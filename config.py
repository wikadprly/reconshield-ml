"""Central configuration for the ReconShield ML service.

Reads from environment variables (a `.env` file is loaded if present) so the
same code runs on your laptop, a lab machine, or a server without edits.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repository root (one level up from this file's directory).
load_dotenv(Path(__file__).resolve().parent / ".env")

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

#: Where raw flow exports (CSV/NetFlow) are dropped for training.
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

#: Cleaned splits, metrics.json, and any cached artefacts.
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

#: Trained model artefacts (*.joblib).
MODELS_DIR = BASE_DIR / "models"

#: Default raw dataset used when `--data` is not passed to the train script.
DEFAULT_RAW_DATA = RAW_DATA_DIR / "flows.csv"

#: Model evaluation metrics consumed by `GET /model/metrics`.
METRICS_PATH = PROCESSED_DATA_DIR / "metrics.json"

# --- Model ----------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "Random Forest Classifier")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")

# RandomForest hyper-parameters (documented in the ML Performance screen).
N_ESTIMATORS = int(os.getenv("RF_N_ESTIMATORS", "100"))
MAX_DEPTH = int(os.getenv("RF_MAX_DEPTH", "12"))
RANDOM_STATE = int(os.getenv("RF_RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("RF_TEST_SIZE", "0.2"))

# --- Database (for sync_metrics_db.py + pipeline.detect) ------------------
#: Same variable name as reconshield-web so one .env can describe the stack.
def _for_psycopg(url: str | None) -> str | None:
    """Strip Prisma-only query params (``schema=...``) from DATABASE_URL.

    reconshield-web's .env carries a Prisma-style URL, but libpq/psycopg2
    rejects unknown URI query parameters. Python only ever talks to the
    database through psycopg2, so the cleanup happens once, here.
    """
    if not url or "?" not in url:
        return url
    base, _, query = url.partition("?")
    keep = [part for part in query.split("&") if part and not part.startswith("schema=")]
    return base + ("?" + "&".join(keep) if keep else "")


DATABASE_URL = _for_psycopg(os.getenv("DATABASE_URL"))

# --- HTTP -----------------------------------------------------------------
API_HOST = os.getenv("ML_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("ML_API_PORT", "8000"))


def ensure_dirs() -> None:
    """Create the data/model directories if they do not exist yet."""
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
