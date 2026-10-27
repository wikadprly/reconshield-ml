"""Feature transformation rules: order, coercion and column dropping."""

import pandas as pd

from preprocess import features as F


def _row(**extra):
    row = {column: 1 for column in F.FEATURE_COLUMNS}
    row.update(extra)
    return row


def test_coerce_frame_returns_every_feature_in_order():
    out = F.coerce_frame(pd.DataFrame([{"prot": 6, "sport": 80}]))

    assert list(out.columns) == F.FEATURE_COLUMNS
    assert out.loc[0, "prot"] == 6
    assert out.loc[0, "sport"] == 80
    # Missing columns are zero-filled instead of raising.
    assert out.loc[0, "dport"] == 0


def test_non_numeric_values_coerce_to_zero():
    out = F.coerce_frame(pd.DataFrame([{"prot": "tcp", "bytes": "512"}]))

    assert out.loc[0, "prot"] == 0
    assert out.loc[0, "bytes"] == 512


def test_select_features_drops_system_and_ip_columns():
    frame = pd.DataFrame(
        [_row(label="recon", src="10.0.0.1", dst="10.0.0.2", exporter="10.10.30.1")]
    )

    out = F.select_features(frame)

    assert list(out.columns) == F.FEATURE_COLUMNS


def test_flow_to_vector_follows_feature_order():
    vector = F.flow_to_vector({"dport": 443, "prot": 6, "packets": 3})

    assert len(vector) == len(F.FEATURE_COLUMNS)
    assert vector[F.FEATURE_COLUMNS.index("dport")] == 443
    assert vector[F.FEATURE_COLUMNS.index("prot")] == 6
    assert vector[F.FEATURE_COLUMNS.index("packets")] == 3


def test_missing_features_reports_the_full_contract():
    assert F.missing_features([1.0, 2.0]) == F.FEATURE_COLUMNS
    assert F.missing_features([0.0] * len(F.FEATURE_COLUMNS)) == []
