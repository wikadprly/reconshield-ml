"""Flow record -> numeric feature vector.

Single source of truth for feature engineering, shared by the training script
and the FastAPI inference endpoint. Keeping it in one place prevents the
classic train/serve skew where the live pipeline transforms fields differently
than the pipeline that produced the model.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

# Column that holds the ground-truth label. Never part of the feature vector.
TARGET_COLUMN = "label"

# Column names that must never be fed to the model even if present.
# `src`/`dst`/`exporter` are IP-related: kept out of the features on purpose so
# the model cannot memorise subnets (same rule as dataset/ta-recon/build_dataset.py).
EXCLUDED_COLUMNS = {
    TARGET_COLUMN,
    "id",
    "timestamp",
    "flow_id",
    "src_ip",
    "dst_ip",
    "recv_ts",
    "exporter",
    "src",
    "dst",
}

#: Canonical label values. The web UI renders exactly these two strings.
LABEL_NORMAL = "Normal"
LABEL_RECON = "Reconnaissance"

# Order matters: this is the feature order the model expects at inference time.
# These are the 16 per-flow features produced by the lab pipeline
# (dataset/ta-recon/build_dataset.py) from NetFlow v5 records — training and
# serving share this list, so any change here must match that pipeline.
FEATURE_COLUMNS: list[str] = [
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

#: Human-readable names for the metrics payload / dashboard charts.
#: Keys must stay in sync with FEATURE_COLUMNS.
FEATURE_LABELS: dict[str, str] = {
    "prot": "Protocol",
    "tos": "Type of Service",
    "tcp_flags": "TCP Flags",
    "sport": "Source Port",
    "dport": "Destination Port",
    "packets": "Packet Count",
    "bytes": "Byte Count",
    "duration_ms": "Flow Duration (ms)",
    "bytes_per_pkt": "Bytes per Packet",
    "pkts_per_sec": "Packet Rate",
    "f_fin": "FIN Flag",
    "f_syn": "SYN Flag",
    "f_rst": "RST Flag",
    "f_psh": "PSH Flag",
    "f_ack": "ACK Flag",
    "f_urg": "URG Flag",
}


def label_for(feature: str) -> str:
    """Display label for a feature column (falls back to the raw name)."""
    return FEATURE_LABELS.get(feature, feature)


def coerce_frame(frame: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    """Return a numeric-only copy of `frame` restricted to FEATURE_COLUMNS.

    By default a partially-shaped record still transforms: missing columns are
    created as zeros and non-numeric values become 0. That tolerance is right
    for a single live flow, but it is exactly what hid the A18 failure during
    training (a renamed/aggregated column silently trained an all-zero model).

    Pass ``strict=True`` on the training path to turn those silent zeros into a
    loud error: every FEATURE_COLUMNS column must be present, and a present
    column must not contain non-numeric text.
    """
    out = pd.DataFrame(index=frame.index)

    if strict:
        missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(
                f"feature frame is missing required columns {missing}; expected the "
                f"{len(FEATURE_COLUMNS)}-feature contract {FEATURE_COLUMNS}. "
                "Refusing to zero-fill silently (this is the A18 failure)."
            )

    for column in FEATURE_COLUMNS:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if strict:
                unparsed = frame[column].notna() & values.isna()
                if bool(unparsed.any()):
                    sample = frame[column][unparsed].iloc[0]
                    raise ValueError(
                        f"feature column '{column}' has a non-numeric value {sample!r}; "
                        "expected a number."
                    )
        else:
            values = pd.Series(0.0, index=frame.index, dtype="float64")
        out[column] = values.fillna(0.0)
    return out


def select_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop excluded/system columns, keep every remaining engineered column.

    Used during training where the CSV may contain extra helper columns. The
    resulting column order becomes the model's `feature_names_in_`.

    Runs ``coerce_frame`` in strict mode: a training frame that is missing any
    of the 16 features must fail loudly instead of training on zeros (A18).
    """
    usable = [c for c in frame.columns if c not in EXCLUDED_COLUMNS]
    return coerce_frame(frame[usable], strict=True) if usable else coerce_frame(frame, strict=True)


def flow_to_vector(flow: Mapping[str, Any]) -> list[float]:
    """Map a single raw flow record (dict) to a list in FEATURE_COLUMNS order."""
    return coerce_frame(pd.DataFrame([dict(flow)])).iloc[0].tolist()


def feature_rows(flows: Iterable[Mapping[str, Any]]) -> list[list[float]]:
    """Convenience wrapper mapping many flow records to vectors."""
    return [flow_to_vector(flow) for flow in flows]


def missing_features(values: Sequence[float]) -> list[str]:
    """Names of features expected by `values`' length, for a readable 400 error."""
    if len(values) != len(FEATURE_COLUMNS):
        return list(FEATURE_COLUMNS)
    return []
