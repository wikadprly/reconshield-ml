"""NetFlow v5 packet simulator for local pipeline testing.

Crafts valid NetFlow v5 datagrams and sends them over UDP so the detection
pipeline (`python -m pipeline.detect`) can be exercised without the lab being
on. Scenarios mirror the real lab topology from dataset/ta-recon/deteksi.md:

  recon  : 10.10.30.5 (attacker) -> 10.10.20.254 (victim), 1-packet SYN
           probes across many ports — the shape build_dataset.py labels "recon".
  normal : 10.10.30.2 -> 8.8.8.8:443, 120 packets / 45 KB over 25 s —
           the shape labeled "normal".

Run from the repository root of reconshield-ml:

    python -m pipeline.simulate_netflow                 # both scenarios
    python -m pipeline.simulate_netflow --scenario recon --repeat 10
"""

from __future__ import annotations

import argparse
import random
import socket
import struct
import time

HEADER_FMT = "!HHIIIIBBH"
RECORD_FMT = "!IIIHHIIIIHHBBBBHHBBH"

SYN, ACK, PSH = 0x02, 0x10, 0x08
TCP = 6

UPTIME_BASE = 1_000_000  # ms since "router uptime" — first/last derive from this


def pack_record(*, src: str, dst: str, sport: int, dport: int, pkts: int,
                byts: int, first: int, last: int, flags: int, prot: int = TCP,
                tos: int = 0) -> bytes:
    to_int = lambda ip: sum(int(o) << (8 * (3 - i)) for i, o in enumerate(ip.split(".")))
    return struct.pack(
        RECORD_FMT,
        to_int(src), to_int(dst), 0,          # src, dst, nexthop
        1, 2,                                  # input, output ifIndex
        pkts, byts, first, last,               # packets, octets, first/last ms
        sport, dport, 0, flags, prot, tos,     # ports, pad1, tcpflags, proto, tos
        0, 0, 24, 24, 0,                       # srcas, dstas, srcmask, dstmask, pad2
    )


def send_batch(sock, target, records, seq: int) -> None:
    payload = struct.pack(
        HEADER_FMT, 5, len(records), UPTIME_BASE,
        int(time.time()), 0, seq, 0, 0, 0,   # version, count, uptime, unix secs/nsecs, seq
    ) + b"".join(records)
    sock.sendto(payload, target)
    time.sleep(0.05)  # keep datagrams well apart like a real exporter


def recon_records(count: int) -> list[bytes]:
    rng = random.Random(42)
    records = []
    for i in range(count):
        records.append(pack_record(
            src="10.10.30.5", dst="10.10.20.254",
            sport=40000 + i, dport=[22, 80, 443, 3389, 8080, 23, 21, 445][i % 8],
            pkts=1, byts=rng.choice([44, 48, 52, 60]),
            first=UPTIME_BASE + i * 10, last=UPTIME_BASE + i * 10 + 1,
            flags=SYN,
        ))
    return records


def normal_records(count: int) -> list[bytes]:
    records = []
    for i in range(count):
        records.append(pack_record(
            src="10.10.30.2", dst="8.8.8.8", sport=51000 + i, dport=443,
            pkts=120, byts=45000,
            first=UPTIME_BASE + i * 1000, last=UPTIME_BASE + i * 1000 + 25_000,
            flags=ACK | PSH,
        ))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Send crafted NetFlow v5 packets")
    parser.add_argument("--target", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2055)
    parser.add_argument("--scenario", choices=["recon", "normal", "both"], default="both")
    parser.add_argument("--repeat", type=int, default=3,
                        help="How many times to replay each scenario batch")
    args = parser.parse_args()

    target = (args.target, args.port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 1
    print(f"[sim] sending NetFlow v5 to {args.target}:{args.port} "
          f"(scenario={args.scenario}, repeat={args.repeat})")

    for _ in range(args.repeat):
        if args.scenario in ("recon", "both"):
            send_batch(sock, target, recon_records(12), seq)
            seq += 1
            print("[sim] sent batch: 12 recon probes 10.10.30.5 -> 10.10.20.254")
        if args.scenario in ("normal", "both"):
            send_batch(sock, target, normal_records(6), seq)
            seq += 1
            print("[sim] sent batch: 6 normal flows 10.10.30.2 -> 8.8.8.8:443")

    sock.close()
    print("[sim] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
