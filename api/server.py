"""FastAPI service exposing the trained Random Forest.

Contract consumed by `reconshield-web/src/services/ml-service.ts`:

    GET  /model/metrics  -> ModelMetrics (see src/types/security.ts)
    POST /predict        -> classification for one flow record

Run from the repository root:

    uvicorn api.server:app --reload --port 8000
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

import config
from preprocess import features as F


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.ensure_dirs()
    try:
        _, name = load_model()
        print(f"[api] loaded model artefact: {name}")
    except FileNotFoundError as exc:
        # Report-only endpoints still work; /predict stays unavailable.
        print(f"[api] {exc}")
    yield


app = FastAPI(
    title="ReconShield ML Service",
    description="Random Forest inference and evaluation metrics for ReconShield.",
    version=config.MODEL_VERSION.lstrip("v"),
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    """A single flow record. Unknown fields are kept so callers can pass raw
    NetFlow rows without us renaming everything first."""

    model_config = ConfigDict(extra="allow")

    def as_flow(self) -> dict[str, Any]:
        merged: dict[str, Any] = dict(self.model_extra or {})
        merged.update(self.model_dump(exclude_unset=True))
        return merged


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


def _latest_model_path() -> Path | None:
    if not config.MODELS_DIR.exists():
        return None
    artefacts = sorted(config.MODELS_DIR.glob("*.joblib"), key=lambda p: p.stat().st_mtime)
    return artefacts[-1] if artefacts else None


def _recorded_artefact_path() -> Path | None:
    """Artefact named by metrics.json, so metrics and inference stay in step.

    `metrics.json` and `models/*.joblib` are written together when training but
    can drift apart afterwards (a copied or deleted file, a hand-edited
    metrics). The recorded name wins whenever that file still exists, so the
    numbers the dashboard shows always describe the model `/predict` uses (T6).
    Only the basename is honoured — the field can never escape `models/`.
    """
    if not config.METRICS_PATH.exists():
        return None
    try:
        recorded = json.loads(config.METRICS_PATH.read_text(encoding="utf-8")).get("artefact")
    except (OSError, json.JSONDecodeError):
        return None
    if not recorded:
        return None
    path = config.MODELS_DIR / Path(str(recorded)).name
    return path if path.exists() else None


@lru_cache(maxsize=1)
def load_model():
    """Load the artefact metrics.json describes, else the newest one.

    Cached for the process lifetime — restart the service after retraining.
    """
    path = _recorded_artefact_path() or _latest_model_path()
    if path is None:
        raise FileNotFoundError(
            "No model artefact found. Run `python -m training.train_rf` first."
        )
    return joblib.load(path), path.name


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        _, name = load_model()
        model_loaded, artefact = True, name
    except FileNotFoundError:
        model_loaded, artefact = False, None
    return {
        "status": "ok",
        "modelLoaded": model_loaded,
        "artefact": artefact,
        "metricsFile": str(config.METRICS_PATH),
        "metricsPresent": config.METRICS_PATH.exists(),
    }


@app.get("/model/metrics")
def model_metrics() -> dict[str, Any]:
    """Latest evaluation run, shaped exactly like the web's `ModelMetrics`."""
    if not config.METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No metrics recorded yet. Train a model with "
                "`python -m training.train_rf` first."
            ),
        )
    try:
        return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"metrics.json unreadable: {exc}") from exc


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Classify one flow record as Normal or Reconnaissance."""
    try:
        model, _ = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    flow = request.as_flow()
    vector = F.flow_to_vector(flow)

    expected = list(getattr(model, "feature_names_in_", F.FEATURE_COLUMNS))
    if len(vector) != len(expected):
        raise HTTPException(status_code=422, detail=f"Expected {len(expected)} features, got {len(vector)}")

    try:
        # Predict on a named DataFrame so sklearn matches the training-time
        # feature names (silences the "no valid feature names" warning).
        frame = pd.DataFrame([vector], columns=expected)
        probabilities = model.predict_proba(frame)[0]
    except Exception as exc:  # noqa: BLE001 - surface model errors as a 422
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc

    classes = list(model.classes_)
    best_index = int(max(range(len(probabilities)), key=lambda i: probabilities[i]))
    return PredictResponse(
        label=str(classes[best_index]),
        confidence=round(float(probabilities[best_index]), 4),
        probabilities={str(c): round(float(p), 4) for c, p in zip(classes, probabilities)},
    )
