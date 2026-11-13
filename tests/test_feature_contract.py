"""The 16-feature contract shared by training, the API and the detection pipeline.

The A18 failure was silent: a renamed feature trained and served without raising
while quietly producing an all-zero model. `coerce_frame(strict=True)` — used by
the training path via `select_features` — now makes a missing or non-numeric
feature a hard error instead. These tests keep the contract loud.

Changing the contract means changing *all* of: `preprocess/features.py`,
`dataset/ta-recon/build_dataset.py`, the model (retrain) and this file.
"""

from pathlib import Path

import pytest

from preprocess import features as F

# Mirrors dataset/ta-recon/build_dataset.py — the lab pipeline that produced
# dataset/ta-recon/data/dataset.csv. Change both or neither.
LAB_FEATURE_COLUMNS = [
    "prot",
    "tos",
    "tcp_flags",
    "sport",
    "dport",
    "packets",
    "bytes",
    "duration_ms",
    "bytes_per_pkt",
    "pkts_per_sec",
    "f_fin",
    "f_syn",
    "f_rst",
    "f_psh",
    "f_ack",
    "f_urg",
]


def test_feature_columns_match_the_lab_contract():
    assert F.FEATURE_COLUMNS == LAB_FEATURE_COLUMNS


def test_every_feature_has_a_display_label():
    assert set(F.FEATURE_LABELS) == set(F.FEATURE_COLUMNS)


def test_labels_and_ip_columns_never_become_features():
    for column in ("label", "src", "dst", "exporter", "id", "timestamp"):
        assert column in F.EXCLUDED_COLUMNS
        assert column not in F.FEATURE_COLUMNS


def test_canonical_labels_are_unchanged():
    # The web UI renders exactly these two strings.
    assert (F.LABEL_NORMAL, F.LABEL_RECON) == ("Normal", "Reconnaissance")
    assert F.TARGET_COLUMN == "label"


def test_training_path_rejects_a_renamed_feature():
    """Renaming a feature must fail loudly on the training path (A18)."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame([{column: 1 for column in LAB_FEATURE_COLUMNS}])
    frame = frame.rename(columns={"pkts_per_sec": "packet_rate"})

    with pytest.raises(ValueError):
        F.select_features(frame)


def test_lab_dataset_carries_every_feature():
    """dataset.csv header must still cover the contract (skipped if absent)."""
    pd = pytest.importorskip("pandas")
    dataset = (
        Path(__file__).resolve().parents[2] / "dataset" / "ta-recon" / "data" / "dataset.csv"
    )
    if not dataset.exists():
        pytest.skip("dataset/ta-recon/data/dataset.csv not present in this checkout")

    columns = set(pd.read_csv(dataset, nrows=0).columns)
    assert set(F.FEATURE_COLUMNS) <= columns, "dataset is missing model features"
    assert F.TARGET_COLUMN in columns, "dataset is missing the label column"
