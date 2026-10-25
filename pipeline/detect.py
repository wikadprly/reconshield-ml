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
    """Writes detections rows; degrades to a warning when the DB is unreachable."""

    def __init__(self, database_url: str, dry_run: bool):
        self.dry_run = dry_run
        self.conn = None if dry_run else psycopg2.connect(database_url, connect_timeout=5)
        if self.conn is not None:
            self.conn.autocommit = True

    def next_id(self) -> str:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id FROM detections "
            "ORDER BY COALESCE(NULLIF(regexp_replace(id, '\\D', '', 'g'), '')::bigint, -1) "
            "DESC LIMIT 1"
        )
        row = cur.fetchone()
        number = int("".join(ch for ch in row[0] if ch.isdigit())) + 1 if row else 1001
        return f"INC-{number}"

    def insert(self, flow: dict, label: str, confidence: float) -> str | None:
        if self.dry_run:
            return None
        detection_id = self.next_id()
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        try:
            cur = self.conn.cursor()
            cur.execute(
                'INSERT INTO detections (id, "timestamp", "sourceIp", "sourcePort", '
                '"destinationIp", "destinationPort", classification, confidence, '
                'action, status, "createdAt", "updatedAt") '
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (detection_id, now, flow["src"], int(flow["sport"]),
                 flow["dst"], int(flow["dport"]), label,
                 round(confidence * 100, 1), "Pending", "Review", now, now),
            )
            return detection_id
        except psycopg2.Error as exc:
            print(f"[detect] DB write failed: {exc.pgerror or exc}", file=sys.stderr)
            return None

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()


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

    store = DetectionStore(config.DATABASE_URL or "", dry_run=args.dry_run)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))

    mode = "DRY-RUN (no DB writes)" if args.dry_run else "DB writes ON"
    print(f"[detect] NetFlow v5 on udp/{args.port} ({args.bind}) | model={artefact} | {mode}")
    print(f"[detect] dedup window={args.dedup_seconds}s | min-confidence={args.min_confidence}"
          + ("" if args.keep_noise_ports else f" | skipping noise ports {sorted(NOISE_PORTS)}"))

    latencies: list[float] = []
    last_seen: dict[str, float] = {}
    counts = {"flows": 0, "skipped_noise": 0, "normal": 0, "detections": 0, "db_errors": 0}

    try:
        while True:
            data, addr = sock.recvfrom(65535)
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
                if now - last_seen.get(flow["src"], 0.0) < args.dedup_seconds:
                    continue
                last_seen[flow["src"]] = now

                detection_id = store.insert(flow, label, confidence)
                counts["detections"] += 1
                stamp = dt.datetime.now().strftime("%H:%M:%S")
                target = detection_id or "dry-run"
                print(f"[detect] {stamp} {label} {confidence * 100:5.1f}% "
                      f"{flow['src']}:{flow['sport']} -> {flow['dst']}:{flow['dport']} "
                      f"prot={flow['prot']} pkts={flow['packets']} bytes={flow['bytes']} "
                      f"dur={flow['duration_ms']}ms -> {target}")
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        store.close()
        if latencies:
            ordered = sorted(latencies)
            p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
            print(f"\n[detect] summary: {counts}")
            print(f"[detect] inference latency over {len(latencies)} flows: "
                  f"mean={sum(latencies) / len(latencies) * 1000:.2f}ms "
                  f"p95={p95 * 1000:.2f}ms max={max(latencies) * 1000:.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
