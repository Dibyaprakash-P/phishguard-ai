"""Shared pytest fixtures.

The suite must run in CI with **no trained model artifact and no API key**, so
fixtures provide a lightweight stub model rather than loading the real one.
That keeps tests fast, hermetic and free of secrets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class StubModel:
    """Deterministic stand-in for the trained classifier.

    Returns a probability derived from a couple of obvious signals so that
    tests can assert on verdict *plumbing* without depending on a real
    artifact or on any particular trained model's behaviour.
    """

    def __init__(self, fixed_probability: float | None = None) -> None:
        self.fixed_probability = fixed_probability
        self.calls: list[int] = []

    def predict_proba(self, frame) -> np.ndarray:
        self.calls.append(len(frame))
        if self.fixed_probability is not None:
            probabilities = np.full(len(frame), self.fixed_probability)
        else:
            probabilities = np.array([
                min(0.95, 0.05 + 0.15 * row["num_suspicious_keywords"] + 0.5 * row["has_ip_host"])
                for _, row in frame.iterrows()
            ])
        return np.column_stack([1 - probabilities, probabilities])


@pytest.fixture
def stub_metadata() -> dict:
    from ml.features import FEATURE_NAMES

    return {
        "model_name": "stub_model",
        "model_version": "0.0.1-test",
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "decision_threshold": 0.5,
        "threshold_policy": "test fixture",
        "risk_bands": {"low": [0, 30], "medium": [31, 70], "high": [71, 100]},
        "top_features": [],
        "dataset": {},
    }


@pytest.fixture
def stub_model() -> StubModel:
    """The stub instance itself, held separately from the singleton.

    The FastAPI lifespan hook loads the real artifact when one exists, which
    would silently replace whatever the fixture installed. Keeping an explicit
    handle lets the client fixtures reinstall the stub *after* startup.
    """
    return StubModel()


@pytest.fixture
def loaded_predictor(stub_metadata, stub_model, monkeypatch):
    """Install the stub model into the predictor singleton for one test."""
    from backend.app.services.predictor import predictor

    model = stub_model
    monkeypatch.setattr(predictor, "_model", model, raising=False)
    monkeypatch.setattr(predictor, "_metadata", stub_metadata, raising=False)
    monkeypatch.setattr(predictor, "_metrics", {}, raising=False)
    monkeypatch.setattr(predictor, "_load_error", None, raising=False)
    return predictor


@pytest.fixture
def unloaded_predictor(monkeypatch):
    """Simulate a deployment where the training pipeline has not been run."""
    from backend.app.services.predictor import predictor

    monkeypatch.setattr(predictor, "_model", None, raising=False)
    monkeypatch.setattr(predictor, "_metadata", {}, raising=False)
    monkeypatch.setattr(predictor, "_metrics", {}, raising=False)
    monkeypatch.setattr(
        predictor, "_load_error", "No model artifact (test fixture).", raising=False
    )
    return predictor


@pytest.fixture
def client(loaded_predictor, stub_model, stub_metadata):
    """TestClient serving the deterministic stub model.

    Startup runs normally, then the stub is reinstalled so tests never depend
    on whether a real artifact happens to exist on the machine.
    """
    from backend.app.main import app

    with TestClient(app) as test_client:
        from backend.app.services.predictor import predictor

        predictor._model = stub_model
        predictor._metadata = stub_metadata
        predictor._metrics = {}
        predictor._load_error = None
        yield test_client


@pytest.fixture
def offline_client(unloaded_predictor):
    """TestClient for a backend with no model available."""
    from backend.app.main import app

    with TestClient(app) as test_client:
        from backend.app.services.predictor import predictor

        predictor._model = None
        predictor._metadata = {}
        predictor._load_error = "No model artifact (test fixture)."
        yield test_client
