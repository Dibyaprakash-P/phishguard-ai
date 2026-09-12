"""Tests for risk interpretation and the prediction service."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.core.config import settings
from backend.app.core.exceptions import ModelNotAvailableError, PredictionError
from backend.app.schemas.analysis import Prediction, RiskLevel
from backend.app.services.feature_extractor import extract
from backend.app.services.predictor import interpret, score_to_risk_level


class TestRiskBands:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (0, RiskLevel.LOW), (30, RiskLevel.LOW),
            (31, RiskLevel.MEDIUM), (70, RiskLevel.MEDIUM),
            (71, RiskLevel.HIGH), (100, RiskLevel.HIGH),
        ],
    )
    def test_score_maps_to_band(self, score, expected):
        assert score_to_risk_level(score) is expected


class TestInterpret:
    def test_low_probability_is_legitimate(self):
        prediction, _, score, confidence = interpret(0.05)
        assert prediction is Prediction.LEGITIMATE
        assert score == 5
        assert confidence == pytest.approx(0.95)

    def test_high_probability_is_phishing(self):
        prediction, level, score, confidence = interpret(0.96)
        assert prediction is Prediction.PHISHING
        assert level is RiskLevel.HIGH
        assert score == 96
        assert confidence == pytest.approx(0.96)

    def test_middle_band_is_suspicious(self):
        midpoint = (settings.suspicious_threshold + settings.phishing_threshold) / 2
        prediction, _, _, confidence = interpret(midpoint)
        assert prediction is Prediction.SUSPICIOUS
        # At the exact midpoint the model is maximally undecided.
        assert confidence == pytest.approx(0.5, abs=1e-6)

    def test_confidence_always_reported_for_the_stated_class(self):
        for probability in (0.01, 0.2, 0.45, 0.6, 0.8, 0.99):
            prediction, _, _, confidence = interpret(probability)
            assert 0.0 <= confidence <= 1.0
            if prediction is Prediction.LEGITIMATE:
                assert confidence == pytest.approx(1 - probability)
            elif prediction is Prediction.PHISHING:
                assert confidence == pytest.approx(probability)

    @pytest.mark.parametrize("probability", [-1.0, 0.0, 1.0, 2.0])
    def test_out_of_range_probabilities_are_clamped(self, probability):
        _, _, score, confidence = interpret(probability)
        assert 0 <= score <= 100
        assert 0.0 <= confidence <= 1.0

    def test_thresholds_are_boundary_inclusive(self):
        assert interpret(settings.phishing_threshold)[0] is Prediction.PHISHING
        assert interpret(settings.suspicious_threshold)[0] is Prediction.SUSPICIOUS


class TestPredictorService:
    def test_unloaded_model_raises_service_error(self, unloaded_predictor):
        with pytest.raises(ModelNotAvailableError):
            unloaded_predictor.predict_one(extract("https://example.com"), "https://example.com")

    def test_prediction_returns_probability(self, loaded_predictor):
        probability = loaded_predictor.predict_one(
            extract("https://example.com"), "https://example.com"
        )
        assert 0.0 <= probability <= 1.0

    def test_batch_prediction_matches_batch_size(self, loaded_predictor):
        urls = ["https://a.com", "https://b.com", "http://192.168.1.1/login"]
        probabilities = loaded_predictor.predict_proba([extract(u) for u in urls], urls)
        assert len(probabilities) == 3
        assert np.all((probabilities >= 0) & (probabilities <= 1))

    def test_mismatched_batch_lengths_are_rejected(self, loaded_predictor):
        with pytest.raises(PredictionError, match="mismatch"):
            loaded_predictor.predict_proba([extract("https://a.com")], ["https://a.com", "https://b.com"])

    def test_model_receives_url_column_and_ordered_features(self, loaded_predictor):
        """Guards the training/serving input contract."""
        from ml.features import FEATURE_NAMES
        from ml.preprocess import MODEL_URL_COLUMN

        captured = {}

        class Recorder:
            def predict_proba(self, frame):
                captured["columns"] = list(frame.columns)
                captured["url"] = frame[MODEL_URL_COLUMN].iloc[0]
                return np.array([[0.4, 0.6]])

        loaded_predictor._model = Recorder()
        loaded_predictor.predict_one(
            extract("https://www.example.com/x"), "https://www.example.com/x"
        )

        assert captured["columns"] == [MODEL_URL_COLUMN, *FEATURE_NAMES]
        # The URL reaches the model canonicalised, exactly as in training.
        assert captured["url"] == "example.com/x"

    def test_inference_failure_is_wrapped(self, loaded_predictor):
        class Broken:
            def predict_proba(self, frame):
                raise RuntimeError("boom")

        loaded_predictor._model = Broken()
        with pytest.raises(PredictionError):
            loaded_predictor.predict_one(extract("https://a.com"), "https://a.com")


class TestFeatureFingerprint:
    """The artifact must not be served by a changed feature extractor.

    `feature_names` matching proves the *columns* line up. It says nothing
    about whether a column still computes the same thing. Fixing
    `num_brand_mentions` so that `paypal.com` stopped counting as PayPal
    impersonation changed every brand-bearing feature vector while leaving all
    52 names identical — a model trained before that change would have been
    served inputs it never saw, with no error anywhere.
    """

    def test_fingerprint_is_deterministic(self):
        from ml.features import feature_fingerprint

        assert feature_fingerprint() == feature_fingerprint()

    def test_fingerprint_changes_when_a_feature_changes(self, monkeypatch):
        import ml.features as features

        before = features.feature_fingerprint()
        # Loosening the brand affix rule to 5 makes "chase" match inside
        # "purchase" again - one probe's num_brand_mentions goes 0 -> 1 while
        # all 52 feature names stay identical. That is precisely the change the
        # name-only contract check cannot see.
        monkeypatch.setattr(features, "_BRAND_AFFIX_MIN_LENGTH", 5)
        assert features.feature_fingerprint() != before

    def test_fingerprint_covers_keyword_changes_too(self, monkeypatch):
        import ml.features as features

        before = features.feature_fingerprint()
        monkeypatch.setattr(features, "SUSPICIOUS_KEYWORDS", ())
        assert features.feature_fingerprint() != before

    def test_load_refuses_a_stale_fingerprint(self, tmp_path, monkeypatch, stub_model):
        import json

        import joblib

        from backend.app.core.config import settings
        from backend.app.services.predictor import PhishingPredictor
        from ml.features import FEATURE_NAMES

        model_path = tmp_path / "model.joblib"
        joblib.dump(stub_model, model_path)
        metadata = tmp_path / "meta.json"
        metadata.write_text(json.dumps({
            "model_name": "stub",
            "feature_names": list(FEATURE_NAMES),
            "feature_fingerprint": "0" * 32,   # not what the extractor produces
        }), encoding="utf-8")

        monkeypatch.setattr(settings, "model_path", str(model_path))
        monkeypatch.setattr(settings, "feature_metadata_path", str(metadata))
        monkeypatch.setattr(settings, "metrics_path", str(tmp_path / "missing.json"))

        predictor = PhishingPredictor()
        assert predictor.load() is False
        assert not predictor.is_loaded
        assert "Feature extractor has changed" in (predictor.load_error or "")

    def test_load_accepts_a_matching_fingerprint(self, tmp_path, monkeypatch, stub_model):
        import json

        import joblib

        from backend.app.core.config import settings
        from backend.app.services.predictor import PhishingPredictor
        from ml.features import FEATURE_NAMES, feature_fingerprint

        model_path = tmp_path / "model.joblib"
        joblib.dump(stub_model, model_path)
        metadata = tmp_path / "meta.json"
        metadata.write_text(json.dumps({
            "model_name": "stub",
            "feature_names": list(FEATURE_NAMES),
            "feature_fingerprint": feature_fingerprint(),
        }), encoding="utf-8")

        monkeypatch.setattr(settings, "model_path", str(model_path))
        monkeypatch.setattr(settings, "feature_metadata_path", str(metadata))
        monkeypatch.setattr(settings, "metrics_path", str(tmp_path / "missing.json"))

        predictor = PhishingPredictor()
        assert predictor.load() is True
        assert predictor.is_loaded
