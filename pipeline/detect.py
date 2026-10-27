"""Automatic detection pipeline: NetFlow v5 -> Random Forest -> PostgreSQL.

Closes the loop from proposal 6.2 (steps 3-6 and 11) that audit finding A3
flagged as missing: flow records arrive over UDP, are transformed with the
*same* `preprocess.features` contract used at training time, classified by the
same model artefact the API serves, and Reconnaissance hits are written to the
`detections` table that `reconshield-web` renders.

Mitigation (A4, RouterOS REST call) is intentionally NOT done here — it is
gated on decision A20 (docs/keputusan-a20-tarpit.md). Detections are written
with action="Pending" / status="Review" until that write path is approved.

Run from the repository root of reconshield-ml:

    python -m pipeline.detect                      # UDP 2055, needs DATABASE_URL
    python -m pipeline.detect --dry-run            # no database writes
    python -m pipeline.detect --port 2055 --min-confidence 0.8

DATABASE_URL is read from the environment or reconshield-ml/.env (same variable
name as reconshield-web, so one .env can describe the stack).
"""

from __future__ import annotations

import argparse
import datetime as dt
import socket
import struct
import sys
import time
from collections import deque

import pandas as pd
import psycopg2

import config
from preprocess import features as F

# NetFlow v5 wire format (identical to dataset/ta-recon/netflow_v5_collector.py).
HEADER_FMT = "!HHIIIIBBH"
HEADER_LEN = struct.calcsize(HEADER_FMT)  # 24
RECORD_FMT = "!IIIHHIIIIHHBBBBHHBBH"
RECORD_LEN = struct.calcsize(RECORD_FMT)  # 48

# TCP flag bits inside a NetFlow v5 record.
FLAG_BITS = {"f_fin": 0x01, "f_syn": 0x02, "f_rst": 0x04,
             "f_psh": 0x08, "f_ack": 0x10, "f_urg": 0x20}

# Ports dropped as noise by dataset/ta-recon/build_dataset.py — the model never
# saw them during training, so predicting on them would be meaningless.
NOISE_PORTS = {2055, 514, 8291}

#: Inference latencies kept for the end-of-run summary. A plain list grew
#: without bound on a long-running pipeline, so this is a rolling window (T5).
LATENCY_WINDOW = 10_000

#: How often expired de-duplication entries are dropped (T5). `last_seen` used
#: to keep one key per source IP for the whole lifetime of the process.
DEDUP_PRUNE_INTERVAL = 60.0


def prune_expired(last_seen: dict[str, float], now: float, window: float) -> dict[str, float]:
    """Drop de-duplication entries whose window has already expired (T5).

    Called on a timer rather than per flow: the dict otherwise keeps one key per
    source IP seen since the process started.
    """
    if not last_seen:
        return last_seen
    horizon = now - window
    return {ip: seen for ip, seen in last_seen.items() if seen > horizon}


def record_outcome(detection_id: str | None, dry_run: bool) -> tuple[bool, str]:
    """Return ``(count_as_detection, log_target)`` for one classified flow (T4).

    A write that did not land in the database must not be reported as a
    detection — otherwise the summary log claims more incidents than the table
    holds. Dry-run writes nothing on purpose, so it still counts.
    """
    if detection_id is None and not dry_run:
        return False, "DB-FAILED"
    return True, detection_id or "dry-run"


def ip_str(x: int) -> str:
    return ".".join(str((x >> s) & 0xFF) for s in (24, 16, 8, 0))


def record_to_flow(rec: tuple) -> dict:
    """Unpack one NetFlow v5 record tuple into a flow dict for the model."""
    (src, dst, _nh, _inp, _outp, pkts, byts, first, last,
     sport, dport, _pad1, tcpflags, prot, tos, _sas, _das,
     _smask, _dmask, _pad2) = rec
    duration = max(last - first, 0)
    flow = {
        "src": ip_str(src),
        "dst": ip_str(dst),
        "sport": sport,
        "dport": dport,
        "prot": prot,
        "tos": tos,
        "tcp_flags": tcpflags,
        "packets": pkts,
        "bytes": byts,
        "duration_ms": duration,
        "bytes_per_pkt": round(byts / pkts, 3) if pkts else 0,
        "pkts_per_sec": round(pkts / (duration / 1000.0), 3) if duration else 0,
    }
    for name, bit in FLAG_BITS.items():
        flow[name] = 1 if tcpflags & bit else 0
    return flow


def predict(model, flow: dict) -> tuple[str, float]:
    """Return (label, confidence 0..1) using the model's training-time columns."""
    vector = F.flow_to_vector(flow)
    expected = list(getattr(model, "feature_names_in_", F.FEATURE_COLUMNS))
    frame = pd.DataFrame([vector], columns=expected)
    probabilities = model.predict_proba(frame)[0]
    classes = list(model.classes_)
    best = int(max(range(len(classes)), key=lambda i: probabilities[i]))
    return str(classes[best]), float(probabilities[best])


class DetectionStore:
    """Writes detections rows and survives the database going away.

    The connection is established lazily and rebuilt on failure: a Postgres
    restart used to leave this object holding a dead cursor for the rest of the
    process, silently discarding every later incident (T1). Failures now retry
    once through a fresh connection and are reported at a bounded rate instead
    of flooding stderr.
    """

    #: Minimum seconds between repeated failure messages.
    ERROR_LOG_INTERVAL = 30.0

    def __init__(self, database_url: str, dry_run: bool, connect=None):
        self.dry_run = dry_run
        self.database_url = database_url
        self._connect = connect or psycopg2.connect
        self.conn = None
        self.errors = 0
        self._last_error_log = 0.0
        if not dry_run:
            # Fail fast at startup, exactly like before, when the DB is absent.
            self._connect_or_raise()

    def _connect_or_raise(self) -> None:
        self.conn = self._connect(self.database_url, connect_timeout=5)
        self.conn.autocommit = True

    def _reconnect(self) -> bool:
        try:
            self._connect_or_raise()
            return True
        except psycopg2.Error as exc:
            self._report_error(f"DB reconnect failed: {exc}")
            return False

    def _healthy(self) -> bool:
        return self.conn is not None and not self.conn.closed

    def _drop(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except psycopg2.Error:
                pass
        self.conn = None

    def _report_error(self, message: str) -> None:
        self.errors += 1
        stamp = time.monotonic()
        if stamp - self._last_error_log >= self.ERROR_LOG_INTERVAL:
            self._last_error_log = stamp
            print(
                f"[detect] {message} - retrying; {self.errors} error(s) so far",
                file=sys.stderr,
            )

    def _run(self, work, what: str):
        """Run `work(conn)` on a live connection, reconnecting once on failure.

        Returns whatever `work` returns, or None when the statement still could
        not be executed after the reconnect.
        """
        for _attempt in (0, 1):
            if not self._healthy() and not self._reconnect():
                return None
            try:
                return work(self.conn)
            except psycopg2.Error as exc:
                self._report_error(f"{what}: {exc.pgerror or exc}")
                self._drop()
        return None

    #: Advisory-lock key that serialises `INC-####` allocation. Arbitrary but
    #: fixed — it only has to be unique inside this application (T3).
    ID_LOCK_KEY = 0x52534348_0001  # "RSCH" + 1

    def insert(self, flow: dict, label: str, confidence: float) -> str | None:
        if self.dry_run:
            return None
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        payload = (
            now, flow["src"], int(flow["sport"]),
            flow["dst"], int(flow["dport"]), label,
            round(confidence * 100, 1), "Pending", "Review", now, now,
        )

        def write(conn) -> str:
            # Read the max id and insert inside ONE transaction holding an
            # advisory lock. The old read-then-insert pair had no lock, so two
            # writers could pick the same number and collide on the primary
            # key (T3). Autocommit is restored even when the statement fails;
            # `_run` then drops and rebuilds the connection.
            previous_autocommit = conn.autocommit
            conn.autocommit = False
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_xact_lock(%s)", (self.ID_LOCK_KEY,))
                    cur.execute(
                        "SELECT id FROM detections "
                        "ORDER BY COALESCE(NULLIF(regexp_replace(id, '\\D', '', 'g'), '')::bigint, -1) "
                        "DESC LIMIT 1"
                    )
                    row = cur.fetchone()
                    number = int("".join(ch for ch in row[0] if ch.isdigit())) + 1 if row else 1001
                    detection_id = f"INC-{number}"
                    cur.execute(
                        'INSERT INTO detections (id, "timestamp", "sourceIp", "sourcePort", '
                        '"destinationIp", "destinationPort", classification, confidence, '
                        'action, status, "createdAt", "updatedAt") '
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (detection_id,) + payload,
                    )
                conn.commit()
                return detection_id
            finally:
                conn.autocommit = previous_autocommit

        return self._run(write, "DB write failed")

    def close(self) -> None:
        self._drop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ReconShield detection pipeline")
    parser.add_argument("--port", type=int, default=2055, help="UDP port for NetFlow v5")
    parser.add_argument("--bind", default="0.0.0.0", help="Bind address")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="Minimum confidence to record a detection (0..1)")
    parser.add_argument("--dedup-seconds", type=int, default=60,
                        help="Suppress repeat detections from the same source IP within this window")
    parser.add_argument("--record-normal", action="store_true",
                        help="Also insert Normal predictions (default: Reconnaissance only)")
    parser.add_argument("--keep-noise-ports", action="store_true",
                        help="Do not skip ports 2055/514/8291 (dropped as noise in training)")
    parser.add_argument("--allow-exporter", action="append", default=None, metavar="IP",
                        help="Accept NetFlow datagrams only from this exporter IP "
                             "(repeatable). Also settable via NETFLOW_ALLOWED_EXPORTERS.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Classify and log, but never touch the database")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.ensure_dirs()

    from api.server import load_model  # shared loader: same artefact the API serves

    try:
        model, artefact = load_model()
    except FileNotFoundError as exc:
        print(f"[detect] {exc}", file=sys.stderr)
        return 1

    if not args.dry_run and not config.DATABASE_URL:
        print("[detect] DATABASE_URL is not set. Put it in reconshield-ml/.env "
              "(same value as reconshield-web/.env) or pass --dry-run.",
              file=sys.stderr)
        return 1

    # NetFlow v5 carries no authentication, so any host that can reach this
    # port could inject incidents. An allowlist keeps the input lab-only (T2).
    allowed = set(args.allow_exporter or []) | set(config.NETFLOW_ALLOWED_EXPORTERS)
    if not allowed:
        print("[detect] WARNING: no exporter allowlist - any host reaching "
              f"udp/{args.port} can inject incidents. Set --allow-exporter or "
              "NETFLOW_ALLOWED_EXPORTERS (see README, section 'Keamanan input NetFlow').")
    else:
        print(f"[detect] exporter allowlist: {sorted(allowed)}")

    try:
        store = DetectionStore(config.DATABASE_URL or "", dry_run=args.dry_run)
    except psycopg2.Error as exc:
        print(f"[detect] cannot reach PostgreSQL: {exc}", file=sys.stderr)
        return 1
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))

    mode = "DRY-RUN (no DB writes)" if args.dry_run else "DB writes ON"
    print(f"[detect] NetFlow v5 on udp/{args.port} ({args.bind}) | model={artefact} | {mode}")
    print(f"[detect] dedup window={args.dedup_seconds}s | min-confidence={args.min_confidence}"
          + ("" if args.keep_noise_ports else f" | skipping noise ports {sorted(NOISE_PORTS)}"))

    # Both used to grow without bound on a long-running pipeline (T5).
    latencies: deque[float] = deque(maxlen=LATENCY_WINDOW)
    last_seen: dict[str, float] = {}
    next_prune = time.monotonic() + DEDUP_PRUNE_INTERVAL
    counts = {"flows": 0, "skipped_noise": 0, "skipped_foreign": 0,
              "normal": 0, "detections": 0, "insert_failed": 0, "db_errors": 0}

    try:
        while True:
            data, addr = sock.recvfrom(65535)
            if allowed and addr[0] not in allowed:
                counts["skipped_foreign"] += 1
                if counts["skipped_foreign"] <= 3:
                    print(f"[detect] ignored datagram from non-allowlisted source {addr[0]}")
                continue
            if len(data) < HEADER_LEN:
                continue
            version, count = struct.unpack("!HH", data[:4])
            if version != 5:
                continue
            n = min(count, (len(data) - HEADER_LEN) // RECORD_LEN)
            off = HEADER_LEN
            for _ in range(n):
                rec = struct.unpack(RECORD_FMT, data[off:off + RECORD_LEN])
                off += RECORD_LEN
                flow = record_to_flow(rec)
                counts["flows"] += 1

                if not args.keep_noise_ports and (
                    flow["sport"] in NOISE_PORTS or flow["dport"] in NOISE_PORTS
                ):
                    counts["skipped_noise"] += 1
                    continue

                started = time.perf_counter()
                label, confidence = predict(model, flow)
                latencies.append(time.perf_counter() - started)

                if label != F.LABEL_RECON:
                    counts["normal"] += 1
                    if not args.record_normal:
                        continue
                if confidence < args.min_confidence:
                    continue

                now = time.monotonic()
                if now >= next_prune:
                    last_seen = prune_expired(last_seen, now, args.dedup_seconds)
                    next_prune = now + DEDUP_PRUNE_INTERVAL
                if now - last_seen.get(flow["src"], 0.0) < args.dedup_seconds:
                    continue
                last_seen[flow["src"]] = now

                counted, target = record_outcome(store.insert(flow, label, confidence), args.dry_run)
                counts["detections" if counted else "insert_failed"] += 1
                stamp = dt.datetime.now().strftime("%H:%M:%S")
                print(f"[detect] {stamp} {label} {confidence * 100:5.1f}% "
                      f"{flow['src']}:{flow['sport']} -> {flow['dst']}:{flow['dport']} "
                      f"prot={flow['prot']} pkts={flow['packets']} bytes={flow['bytes']} "
                      f"dur={flow['duration_ms']}ms -> {target}")
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        store.close()
        counts["db_errors"] = store.errors
        if latencies:
            ordered = sorted(latencies)
            p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
            print(f"\n[detect] summary: {counts}")
            print(f"[detect] inference latency over the last {len(latencies)} flows: "
                  f"mean={sum(latencies) / len(latencies) * 1000:.2f}ms "
                  f"p95={p95 * 1000:.2f}ms max={max(latencies) * 1000:.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
