"""Model loading, inference and risk scoring.

The model artifact is loaded **once** at application startup and held in a
module-level singleton. Nothing in the request path touches disk or retrains
anything; a prediction is a feature extraction plus one ``predict_proba`` call.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.app.core.config import settings
from backend.app.core.exceptions import ModelNotAvailableError, PredictionError
from backend.app.schemas.analysis import Prediction, RiskLevel
from ml.features import FEATURE_NAMES, canonicalize_for_features
from ml.preprocess import MODEL_URL_COLUMN
from ml.reputation import known_good_domain

logger = logging.getLogger(__name__)


class PhishingPredictor:
    """Thread-safe wrapper around the trained classifier."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._metadata: dict[str, Any] = {}
        self._metrics: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._load_error: str | None = None

    # ------------------------------------------------------------ lifecycle
    def load(self) -> bool:
        """Load the artifact from disk. Returns True on success.

        A missing artifact is *not* a fatal error: the API still starts and
        reports ``trained: false`` so the UI can tell the user to run the
        training pipeline, rather than the whole service failing to boot.
        """
        with self._lock:
            self._load_error = None
            model_path = Path(settings.model_path)
            if not model_path.is_file():
                self._load_error = (
                    f"No model artifact at {model_path}. "
                    "Run: python -m ml.train"
                )
                logger.warning(self._load_error)
                self._model = None
                return False

            try:
                started = time.perf_counter()
                self._model = joblib.load(model_path)
                elapsed = (time.perf_counter() - started) * 1000
            except Exception as exc:
                self._load_error = f"Failed to deserialize model artifact: {exc}"
                logger.exception("Model load failed")
                self._model = None
                return False

            self._metadata = self._read_json(settings.feature_metadata_path)
            self._metrics = self._read_json(settings.metrics_path)

            expected = self._metadata.get("feature_names")
            if expected and list(expected) != list(FEATURE_NAMES):
                # Refuse to serve a model trained on a different feature
                # contract - silently misaligned columns would produce
                # confidently wrong predictions.
                self._load_error = (
                    "Model/feature contract mismatch: the artifact was trained on a "
                    "different feature set than ml.features currently defines. Retrain "
                    "with: python -m ml.train"
                )
                logger.error(self._load_error)
                self._model = None
                return False

            logger.info(
                "Loaded model '%s' v%s (%d features) in %.0f ms",
                self.model_name, self.model_version, len(FEATURE_NAMES), elapsed,
            )
            return True

    @staticmethod
    def _read_json(path: Path | str) -> dict[str, Any]:
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            return {}

    # ----------------------------------------------------------- properties
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def model_name(self) -> str:
        return str(self._metadata.get("model_name", "unknown"))

    @property
    def model_version(self) -> str:
        return str(self._metadata.get("model_version", "0.0.0"))

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    @property
    def threshold(self) -> float:
        """Decision threshold chosen on the validation split during training."""
        try:
            return float(self._metadata.get("decision_threshold", 0.5))
        except (TypeError, ValueError):
            return 0.5

    # ------------------------------------------------------------ inference
    def predict_proba(
        self, features_list: list[dict[str, float]], urls: list[str]
    ) -> np.ndarray:
        """Return P(phishing) for a batch of (features, url) pairs.

        The frame built here is byte-for-byte the one produced by
        ``ml.preprocess.build_model_frame`` during training: the canonical URL
        text followed by the ordered numeric features. Every candidate model
        accepts this same contract, so the selected model - numeric-only or
        hybrid - is served identically.
        """
        if self._model is None:
            raise ModelNotAvailableError(
                "The detection model is not available.",
                self._load_error or "Model artifact has not been loaded.",
            )
        if len(features_list) != len(urls):
            raise PredictionError("Feature/URL batch length mismatch.")
        try:
            matrix = pd.DataFrame(
                [[f[name] for name in FEATURE_NAMES] for f in features_list],
                columns=list(FEATURE_NAMES),
            ).astype("float32")
            matrix.insert(0, MODEL_URL_COLUMN, [canonicalize_for_features(u) for u in urls])
            return self._model.predict_proba(matrix)[:, 1]
        except Exception as exc:
            logger.exception("Inference failed")
            raise PredictionError("Prediction failed for the submitted input.", str(exc)) from exc

    def predict_one(self, features: dict[str, float], url: str) -> float:
        """Return P(phishing) for a single URL."""
        return float(self.predict_proba([features], [url])[0])


# --------------------------------------------------------------------------
# Reputation prior
# --------------------------------------------------------------------------

def apply_reputation(probability: float, url: str) -> tuple[float, str | None]:
    """Clamp ``probability`` for URLs served from a known-good registrable domain.

    Returns ``(probability, matched_domain)``; ``matched_domain`` is ``None``
    when no reputation rule applied, which is the case for every URL off the
    curated list in :mod:`ml.reputation`.

    The clamp is one-directional - ``min`` of the model output and the ceiling -
    so reputation can only ever lower a score, never raise one. A known-good
    domain the model already scored confidently legitimate keeps that lower
    number rather than being pulled up to the ceiling.

    This exists because a URL-only classifier cannot read a bare domain: there
    is no lure vocabulary, no depth, no entropy to score, so the model returns
    something near its prior and well-known sites land mid-band. See
    :mod:`ml.reputation` for the measurements and the bypass guards.
    """
    if settings.reputation_ceiling >= 1.0:
        return probability, None
    domain = known_good_domain(url)
    if domain is None:
        return probability, None
    return min(probability, settings.reputation_ceiling), domain


# --------------------------------------------------------------------------
# Risk interpretation
# --------------------------------------------------------------------------

def score_to_risk_level(risk_score: int) -> RiskLevel:
    """Map a 0-100 risk score onto a configurable band."""
    if risk_score <= settings.risk_low_max:
        return RiskLevel.LOW
    if risk_score <= settings.risk_medium_max:
        return RiskLevel.MEDIUM
    return RiskLevel.HIGH


def interpret(
    probability: float, phishing_threshold: float | None = None
) -> tuple[Prediction, RiskLevel, int, float]:
    """Turn P(phishing) into ``(prediction, risk_level, risk_score, confidence)``.

    Three verdicts are produced from a binary classifier by carving a
    ``suspicious`` band out of the middle of the probability range, where the
    model is least certain.

    The upper boundary defaults to the model's **tuned operating point** - the
    threshold selected on the validation split to satisfy the precision floor.
    Using any other cut-off would mean reporting a verdict at a precision the
    model was never measured at. ``USE_MODEL_THRESHOLD=false`` pins it to the
    configured ``phishing_threshold`` instead.

    Both boundaries are returned with every API response, so a risk score is
    never an unexplained number.

    ``confidence`` is the model's confidence in the class it reported: ``p``
    for a phishing verdict, ``1 - p`` for a legitimate one. In the
    ``suspicious`` band, where neither class is asserted, it reports how far
    the probability sits from the middle of that band.
    """
    probability = float(min(max(probability, 0.0), 1.0))
    risk_score = int(round(probability * 100))
    risk_level = score_to_risk_level(risk_score)

    upper = phishing_threshold if phishing_threshold is not None else settings.phishing_threshold
    # The suspicious band must sit below the phishing boundary; a tuned
    # threshold can legitimately fall under the configured lower bound.
    lower = min(settings.suspicious_threshold, upper)

    if probability >= upper:
        return Prediction.PHISHING, risk_level, risk_score, probability
    if probability >= lower:
        midpoint = (lower + upper) / 2
        span = max(upper - lower, 1e-9)
        confidence = 0.5 + abs(probability - midpoint) / span
        return Prediction.SUSPICIOUS, risk_level, risk_score, min(confidence, 1.0)
    return Prediction.LEGITIMATE, risk_level, risk_score, 1.0 - probability


def verdict_threshold() -> float:
    """The phishing boundary currently in force."""
    if settings.use_model_threshold and predictor.is_loaded:
        return predictor.threshold
    return settings.phishing_threshold


# --------------------------------------------------------------------------
# Singleton
# --------------------------------------------------------------------------

predictor = PhishingPredictor()
