"""Train the RandomForest classifier for ReconShield.

Reads a flow dataset from CSV, engineers features with the shared
`preprocess.features` helpers, splits train/test (session-aware by source IP,
falling back to temporal then stratified random), fits a RandomForest, then
writes:

  * `models/<MODEL_NAME>_<MODEL_VERSION>.joblib` — the artefact served at inference
  * `data/processed/metrics.json`                — consumed by GET /model/metrics

Usage:
    python -m training.train_rf
    python -m training.train_rf --data data/raw/flows.csv --target label
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

import config
from preprocess import features as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ReconShield Random Forest")
    parser.add_argument(
        "--data",
        type=Path,
        default=config.DEFAULT_RAW_DATA,
        help="Path to the raw flow CSV (default: data/raw/flows.csv)",
    )
    parser.add_argument(
        "--target",
        default=F.TARGET_COLUMN,
        help="Name of the label column (default: label)",
    )
    parser.add_argument(
        "--version",
        default=config.MODEL_VERSION,
        help="Model version tag written into metrics + filename",
    )
    return parser.parse_args()


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )


#: Columns that identify one reconnaissance session. Every flow from the same
#: source belongs to the same session, so keeping a source on one side of the
#: split stops the model from memorising an attacker it has already seen.
SESSION_COLUMNS = ("src", "src_ip", "sourceIp")

#: Columns that order flows in time (used for a temporal fallback split).
TIME_COLUMNS = ("recv_ts", "timestamp", "time")


def _both_classes(labels: pd.Series) -> bool:
    return labels.nunique() >= 2


def _stratified_group_split(
    groups: np.ndarray, labels: np.ndarray, test_size: float, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole groups, keeping both classes on both sides.

    Each class contributes roughly ``test_size`` of its source groups to the
    test set (at least one, at most all-but-one), so a class whose flows all
    come from distinct sources is still represented in train and test. Groups
    never appear on both sides, so no source is memorised across the split.
    """
    rng = np.random.default_rng(random_state)
    test_mask = np.zeros(len(groups), dtype=bool)
    for klass in pd.unique(labels):
        members = np.array(pd.unique(groups[labels == klass]))
        rng.shuffle(members)
        n_test = min(max(int(round(len(members) * test_size)), 1), len(members) - 1)
        if n_test <= 0:
            continue
        test_mask |= np.isin(groups, members[:n_test])
    return np.where(~test_mask)[0], np.where(test_mask)[0]


def split_train_test(
    frame: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    """Split X/y without leaking one source IP across train and test.

    A plain `train_test_split` scatters flows from the same source into both
    sides, letting the model memorise the attacker and report an optimistic
    score. Preference order:

    1. **Session-aware** (stratified group split on the source-IP column).
    2. **Temporal** (oldest rows train, newest rows test) when a timestamp exists.
    3. **Stratified random** as the last resort.

    Strategies 1-2 can leave a side with a single class (e.g. every
    reconnaissance session sits at the end of the capture). When that happens we
    fall back to the stratified random split so training still runs, and the
    returned strategy string records which one was used.
    """
    n = len(X)

    session_column = next((c for c in SESSION_COLUMNS if c in frame.columns), None)
    if session_column is not None and n:
        groups = frame[session_column].astype(str).to_numpy()
        if len(set(groups)) >= 2 and _both_classes(y):
            train_idx, test_idx = _stratified_group_split(
                groups, y.to_numpy(), test_size, random_state
            )
            if len(train_idx) and len(test_idx) and _both_classes(y.iloc[train_idx]) and _both_classes(y.iloc[test_idx]):
                return (
                    X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx],
                    f"session-aware (by {session_column})",
                )

    time_column = next((c for c in TIME_COLUMNS if c in frame.columns), None)
    if time_column is not None and n:
        order = np.argsort(frame[time_column].to_numpy(), kind="stable")
        cut = int(n * (1 - test_size))
        train_idx, test_idx = order[:cut], order[cut:]
        if len(train_idx) and len(test_idx) and _both_classes(y.iloc[train_idx]) and _both_classes(y.iloc[test_idx]):
            return (
                X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx],
                f"temporal (by {time_column})",
            )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    return X_train, X_test, y_train, y_test, "stratified random (no usable session/time column)"


def normalise_labels(series: pd.Series) -> pd.Series:
    """Map arbitrary label spellings onto the two canonical classes."""
    mapping = {
        "normal": F.LABEL_NORMAL,
        "benign": F.LABEL_NORMAL,
        "legit": F.LABEL_NORMAL,
        "reconnaissance": F.LABEL_RECON,
        "recon": F.LABEL_RECON,
        "scan": F.LABEL_RECON,
        "attack": F.LABEL_RECON,
    }
    cleaned = series.astype(str).str.strip().str.lower()
    mapped = cleaned.map(mapping)
    if mapped.isna().any():
        unknown = sorted(mapped[mapped.isna()].unique())
        raise SystemExit(
            f"Unrecognised labels in target column: {unknown}. "
            f"Expected one of: {sorted(set(mapping.values()))} "
            f"(or their aliases)."
        )
    return mapped


def evaluate(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    version: str,
    dataset: str,
) -> dict:
    y_pred = model.predict(X_test)
    labels = [F.LABEL_NORMAL, F.LABEL_RECON]
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    # With labels ordered [Normal, Recon]:
    #   [0,0] True Normal, [0,1] False Threat (Normal misread as Recon),
    #   [1,0] Missed Threat, [1,1] True Threat
    true_normal, false_threat, missed_threat, true_threat = (int(x) for x in matrix.ravel())

    # Importances are normalised fractions (sum to 1.0) — the same scale the
    # dashboard's mock data uses, so charts stay comparable across sources.
    importances = sorted(
        (
            {"feature": F.label_for(name), "importance": round(float(score), 4)}
            for name, score in zip(model.feature_names_in_, model.feature_importances_)
        ),
        key=lambda row: row["importance"],
        reverse=True,
    )

    return {
        "modelName": config.MODEL_NAME,
        "version": version,
        # Stored as percentages (0-100), matching the seeded example values.
        "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
        "precision": round(float(precision_score(y_test, y_pred, average="binary", pos_label=F.LABEL_RECON)) * 100, 2),
        "recall": round(float(recall_score(y_test, y_pred, average="binary", pos_label=F.LABEL_RECON)) * 100, 2),
        "f1Score": round(float(f1_score(y_test, y_pred, average="binary", pos_label=F.LABEL_RECON)) * 100, 2),
        "confusionMatrix": {
            "trueNormal": true_normal,
            "falseThreat": false_threat,
            "missedThreat": missed_threat,
            "trueThreat": true_threat,
        },
        "featureImportance": importances,
        "dataset": dataset,
        "samples": int(len(X_test)),
        "evaluatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def main() -> int:
    args = parse_args()
    config.ensure_dirs()

    if not args.data.exists():
        print(f"[train] dataset not found: {args.data}", file=sys.stderr)
        print("[train] drop a flow CSV into data/raw/ and re-run.", file=sys.stderr)
        return 1

    print(f"[train] reading {args.data}")
    frame = pd.read_csv(args.data)

    if args.target not in frame.columns:
        print(f"[train] target column '{args.target}' missing; found {list(frame.columns)}", file=sys.stderr)
        return 1

    y = normalise_labels(frame[args.target])
    X = F.select_features(frame.drop(columns=[args.target]))

    if X.empty or X.shape[1] == 0:
        print("[train] no usable feature columns after dropping system columns", file=sys.stderr)
        return 1

    if y.nunique() < 2:
        print(f"[train] target has a single class ({y.unique()}); cannot train", file=sys.stderr)
        return 1

    X_train, X_test, y_train, y_test, strategy = split_train_test(
        frame, X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    print(f"[train] {len(X_train)} train / {len(X_test)} test rows, {X.shape[1]} features")
    print(f"[train] split strategy: {strategy}")

    model = build_model()
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test, args.version, dataset=args.data.stem)

    artifact = config.MODELS_DIR / f"{config.MODEL_NAME.replace(' ', '_')}_{args.version}.joblib"
    # Record which file these numbers belong to: api.server.load_model() reads it
    # back, so the dashboard's metrics and the model behind /predict cannot drift
    # apart (T6).
    metrics["artefact"] = artifact.name
    joblib.dump(model, artifact)
    print(f"[train] saved model -> {artifact}")

    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[train] saved metrics -> {config.METRICS_PATH}")

    print(
        "[train] accuracy={accuracy} precision={precision} recall={recall} f1={f1}".format(
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1Score"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
