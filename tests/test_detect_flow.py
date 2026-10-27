"""NetFlow v5 record -> flow dict -> the 16 model features."""

import struct

from pipeline.detect import (
    NOISE_PORTS,
    RECORD_FMT,
    prune_expired,
    record_outcome,
    record_to_flow,
)
from preprocess import features as F

SYN, ACK = 0x02, 0x10


def _record(
    src="10.10.30.5",
    dst="10.10.20.254",
    sport=40000,
    dport=22,
    pkts=1,
    byts=44,
    first=1_000_000,
    last=1_000_002,
    flags=SYN,
    prot=6,
    tos=0,
) -> tuple:
    """One packed v5 record, exactly as `main()` unpacks it."""
    to_int = lambda ip: sum(int(o) << (8 * (3 - i)) for i, o in enumerate(ip.split(".")))
    payload = struct.pack(
        RECORD_FMT,
        to_int(src), to_int(dst), 0,   # src, dst, nexthop
        1, 2,                          # input, output ifIndex
        pkts, byts, first, last,       # packets, octets, first/last ms
        sport, dport, 0, flags, prot, tos,
        0, 0, 24, 24, 0,
    )
    return struct.unpack(RECORD_FMT, payload)


def test_record_to_flow_exposes_every_model_feature():
    flow = record_to_flow(_record())

    assert set(F.FEATURE_COLUMNS) <= set(flow)
    assert flow["src"] == "10.10.30.5"
    assert flow["dst"] == "10.10.20.254"
    assert flow["sport"] == 40000
    assert flow["dport"] == 22


def test_tcp_flags_are_split_into_binary_columns():
    flow = record_to_flow(_record(flags=SYN | ACK))

    assert flow["tcp_flags"] == SYN | ACK
    assert flow["f_syn"] == 1
    assert flow["f_ack"] == 1
    assert flow["f_fin"] == 0
    assert flow["f_rst"] == 0


def test_derived_rates_are_computed_from_duration():
    flow = record_to_flow(_record(pkts=10, byts=1000, first=100, last=2100))

    assert flow["duration_ms"] == 2000
    assert flow["bytes_per_pkt"] == 100.0
    assert flow["pkts_per_sec"] == 5.0  # 10 packets over 2 seconds


def test_zero_duration_and_zero_packets_do_not_divide_by_zero():
    flow = record_to_flow(_record(pkts=0, byts=0, first=100, last=50))

    assert flow["duration_ms"] == 0
    assert flow["bytes_per_pkt"] == 0
    assert flow["pkts_per_sec"] == 0


def test_noise_ports_match_the_dataset_pipeline():
    # build_dataset.py drops these during training, so the pipeline must too.
    assert NOISE_PORTS == {2055, 514, 8291}


def test_prune_expired_keeps_only_entries_inside_the_window():
    """T5: last_seen must not keep one key per IP for the process lifetime."""
    now = 1_000.0
    seen = {"fresh": now - 5, "stale": now - 120, "edge": now - 60}

    pruned = prune_expired(seen, now, window=60)

    assert "fresh" in pruned
    assert "stale" not in pruned
    assert "edge" not in pruned  # exactly on the horizon is already expired
    assert prune_expired({}, now, 60) == {}


def test_a_failed_write_is_not_counted_as_a_detection():
    """T4: the summary must not claim incidents that never reached the table."""
    assert record_outcome("INC-1058", dry_run=False) == (True, "INC-1058")
    # Dry-run writes nothing on purpose, so it still counts as a detection.
    assert record_outcome(None, dry_run=True) == (True, "dry-run")
    assert record_outcome(None, dry_run=False) == (False, "DB-FAILED")
