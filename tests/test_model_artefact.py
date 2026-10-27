"""metrics.json and the model file must point at each other (audit finding T6).

`load_model()` used to pick the newest *.joblib by mtime while `/model/metrics`
read metrics.json, so a copied or deleted artefact could leave the dashboard
reporting one model while inference ran another.
"""

import json
import os

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier

import api.server as server
import config


def _tiny_model():
    model = RandomForestClassifier(n_estimators=3, random_state=0)
    model.fit([[0.0], [1.0]], ["Normal", "Reconnaissance"])
    return model


def _dump(model, path, mtime):
    joblib.dump(model, path)
    os.utime(path, (mtime, mtime))


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    models = tmp_path / "models"
    processed = tmp_path / "processed"
    models.mkdir()
    processed.mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(config, "METRICS_PATH", processed / "metrics.json")
    server.load_model.cache_clear()
    yield models, processed
    server.load_model.cache_clear()


def test_recorded_artefact_wins_over_the_newer_file(dirs):
    models, processed = dirs
    recorded = models / "Random_Forest_Classifier_v1.1.0.joblib"
    newcomer = models / "Random_Forest_Classifier_v1.2.0.joblib"
    _dump(_tiny_model(), recorded, 1_700_000_000)
    _dump(_tiny_model(), newcomer, 1_800_000_000)  # clearly the newer file
    (processed / "metrics.json").write_text(json.dumps({"artefact": recorded.name}))

    _model, name = server.load_model()

    assert name == recorded.name


def test_stale_recorded_name_falls_back_to_the_newest_file(dirs):
    models, processed = dirs
    newest = models / "Random_Forest_Classifier_v1.1.0.joblib"
    _dump(_tiny_model(), newest, 1_700_000_000)
    (processed / "metrics.json").write_text(json.dumps({"artefact": "deleted.joblib"}))

    _model, name = server.load_model()

    assert name == newest.name


def test_metrics_without_artefact_keep_the_old_mtime_behaviour(dirs):
    models, _processed = dirs
    older = models / "Random_Forest_Classifier_v1.0.0.joblib"
    newer = models / "Random_Forest_Classifier_v1.1.0.joblib"
    _dump(_tiny_model(), older, 1_700_000_000)
    _dump(_tiny_model(), newer, 1_800_000_000)

    _model, name = server.load_model()

    assert name == newer.name
