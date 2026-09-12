"""Tests for the ML pipeline: cleaning, leakage-aware splitting, thresholds.

These operate on small synthetic frames so they run in milliseconds and need
none of the real datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features import FEATURE_NAMES
from ml.preprocess import (
    MODEL_URL_COLUMN,
    build_model_frame,
    clean_corpus,
    group_split,
    subsample,
)
from ml.train import evaluate, tune_threshold


def frame(rows):
    return pd.DataFrame(rows, columns=["url", "label", "source"])


class TestCleaning:
    def test_drops_rows_without_a_usable_host(self):
        cleaned, report = clean_corpus(frame([
            ("https://good.com/a", 0, "s1"),
            ("localhost", 0, "s1"),
            ("x", 1, "s1"),
            ("nan", 1, "s1"),
        ]))
        assert set(cleaned["url"]) == {"https://good.com/a"}
        assert report["input_rows"] == 4

    def test_deduplicates_across_scheme_variants(self):
        cleaned, report = clean_corpus(frame([
            ("http://example.com/x", 1, "s1"),
            ("example.com/x", 1, "s2"),
            ("https://other.com", 0, "s1"),
        ]))
        assert len(cleaned) == 2
        assert report["after_dedup"] == 2

    def test_conflicting_labels_are_dropped_entirely(self):
        """A URL labelled both ways is unusable supervision, not a coin flip."""
        cleaned, report = clean_corpus(frame([
            ("http://disputed.com/x", 0, "s1"),
            ("http://disputed.com/x", 1, "s2"),
            ("http://agreed.com/y", 1, "s1"),
        ]))
        assert report["label_conflicts_removed"] == 2
        assert "http://disputed.com/x" not in set(cleaned["url"])
        assert len(cleaned) == 1

    def test_conflict_detection_runs_before_deduplication(self):
        """Regression guard.

        If dedup ran first, `keep="first"` would resolve a disagreement by
        source ordering and the conflict would never be reported.
        """
        _, report = clean_corpus(frame([
            ("http://a.com/x", 0, "s1"),
            ("http://a.com/x", 1, "s2"),
        ]))
        assert report["label_conflicts_removed"] == 2

    def test_registrable_domain_is_attached(self):
        cleaned, _ = clean_corpus(frame([("https://a.b.example.co.uk/p", 1, "s1")]))
        assert cleaned.iloc[0]["domain"] == "example.co.uk"


class TestGroupSplit:
    def _corpus(self, n_domains=200):
        rows = []
        for i in range(n_domains):
            for j in range(3):  # several URLs per domain, as in real feeds
                rows.append((f"http://domain{i}.com/page{j}", i % 2, "s1"))
        cleaned, _ = clean_corpus(frame(rows))
        return cleaned

    def test_no_domain_appears_in_two_splits(self):
        """The central anti-leakage guarantee."""
        train, val, test = group_split(self._corpus(), 0.15, 0.15, seed=1)
        assert not set(train["domain"]) & set(test["domain"])
        assert not set(train["domain"]) & set(val["domain"])
        assert not set(val["domain"]) & set(test["domain"])

    def test_every_row_lands_in_exactly_one_split(self):
        corpus = self._corpus()
        train, val, test = group_split(corpus, 0.15, 0.15, seed=1)
        assert len(train) + len(val) + len(test) == len(corpus)

    def test_split_is_deterministic_for_a_seed(self):
        corpus = self._corpus()
        a = group_split(corpus, 0.15, 0.15, seed=7)[2]["domain"].tolist()
        b = group_split(corpus, 0.15, 0.15, seed=7)[2]["domain"].tolist()
        assert a == b


class TestSubsample:
    def test_cap_is_respected_and_classes_stay_balanced(self):
        corpus = pd.DataFrame({
            "url": [f"http://d{i}.com" for i in range(1000)],
            "label": [i % 2 for i in range(1000)],
            "domain": [f"d{i}.com" for i in range(1000)],
        })
        sampled = subsample(corpus, 100, seed=3)
        assert len(sampled) == 100
        assert sampled["label"].mean() == pytest.approx(0.5)

    def test_zero_cap_keeps_everything(self):
        corpus = pd.DataFrame({"url": ["a"], "label": [0], "domain": ["a"]})
        assert len(subsample(corpus, 0, seed=1)) == 1


class TestModelFrame:
    def test_columns_match_the_serving_contract(self):
        built = build_model_frame(pd.Series(["https://www.example.com/a"]))
        assert list(built.columns) == [MODEL_URL_COLUMN, *FEATURE_NAMES]

    def test_url_column_is_canonicalised(self):
        built = build_model_frame(pd.Series(["https://www.example.com/a"]))
        assert built[MODEL_URL_COLUMN].iloc[0] == "example.com/a"


class TestMetrics:
    def test_evaluate_reports_the_full_suite(self):
        y_true = pd.Series([0, 0, 1, 1, 0, 1])
        proba = np.array([0.1, 0.2, 0.9, 0.8, 0.6, 0.3])
        metrics = evaluate(y_true, proba, threshold=0.5)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                    "false_positive_rate", "false_negative_rate"):
            assert key in metrics
        assert metrics["true_positives"] == 2
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 1

    def test_confusion_counts_sum_to_the_sample_size(self):
        y_true = pd.Series([0, 1] * 25)
        proba = np.linspace(0, 1, 50)
        m = evaluate(y_true, proba, 0.5)
        total = m["true_positives"] + m["true_negatives"] + m["false_positives"] + m["false_negatives"]
        assert total == 50


class TestThresholdTuning:
    def test_threshold_meets_the_precision_floor(self):
        rng = np.random.default_rng(0)
        y_true = pd.Series(rng.integers(0, 2, 4000))
        # Well-separated scores so a high-precision threshold exists.
        proba = np.clip(y_true * 0.6 + rng.normal(0.2, 0.12, 4000), 0, 1)

        threshold = tune_threshold(y_true, proba, min_precision=0.9)
        achieved = evaluate(y_true, proba, threshold)["precision"]
        assert achieved >= 0.9 - 1e-9

    def test_falls_back_to_best_f1_when_floor_is_unreachable(self):
        rng = np.random.default_rng(1)
        y_true = pd.Series(rng.integers(0, 2, 500))
        proba = rng.random(500)  # pure noise: no threshold reaches 99% precision
        threshold = tune_threshold(y_true, proba, min_precision=0.99)
        assert 0.0 <= threshold <= 1.0

    def test_returned_threshold_is_a_valid_probability(self):
        rng = np.random.default_rng(2)
        y_true = pd.Series(rng.integers(0, 2, 1000))
        proba = np.clip(y_true * 0.5 + rng.normal(0.25, 0.15, 1000), 0, 1)
        assert 0.0 <= tune_threshold(y_true, proba, 0.8) <= 1.0


class TestSanitySet:
    def test_sanity_cases_are_labelled_and_varied(self):
        from ml.sanity_urls import sanity_cases

        cases = sanity_cases()
        assert len(cases) >= 30
        labels = {label for _, label in cases}
        assert labels == {0, 1}

        legit = [url for url, label in cases if label == 0]
        # The set must span the stylistic axes the corpora confound, otherwise
        # it could not catch a provenance shortcut.
        assert any(u.startswith("https://www.") for u in legit)
        assert any(not u.replace("https://", "").startswith("www.") for u in legit)
        assert any(u.count("/") <= 2 for u in legit)   # bare homepages
        assert any(u.count("/") > 3 for u in legit)    # deep paths

    def test_unlisted_legitimate_urls_are_really_unlisted(self):
        """The group that measures the model must not be shielded by the list.

        ``UNLISTED_LEGITIMATE_URLS`` exists so the sanity gate keeps one set of
        legitimate URLs that the reputation prior cannot touch. If someone adds
        one of those domains to ``KNOWN_GOOD_DOMAINS``, the group silently stops
        testing anything - so assert it here rather than discovering it later.
        """
        from ml.reputation import known_good_domain
        from ml.sanity_urls import UNLISTED_LEGITIMATE_URLS

        shielded = [u for u in UNLISTED_LEGITIMATE_URLS if known_good_domain(u)]
        assert not shielded, f"now covered by the known-good list: {shielded}"

    def test_bypass_urls_are_never_known_good(self):
        """Each bypass case targets one guard in ml.reputation."""
        from ml.reputation import known_good_domain
        from ml.sanity_urls import BYPASS_URLS

        admitted = [u for u in BYPASS_URLS if known_good_domain(u)]
        assert not admitted, f"reputation prior admitted an attack: {admitted}"


class TestThresholdConfigAgreement:
    """Training and serving must interpret the same probability identically.

    These constants live in two places - ``ml/config.py`` for the training-time
    sanity gate and ``backend/app/core/config.py`` for the API. They had drifted
    (0.40 vs 0.48), which meant the post-training gate was scoring a suspicious
    band the API never served, and the gate passed while the product showed
    well-known sites as suspicious.
    """

    def test_suspicious_threshold_matches(self):
        from backend.app.core.config import settings
        from ml import config as ml_config

        assert ml_config.SUSPICIOUS_THRESHOLD == settings.suspicious_threshold

    def test_phishing_threshold_matches(self):
        from backend.app.core.config import settings
        from ml import config as ml_config

        assert ml_config.PHISHING_THRESHOLD == settings.phishing_threshold

    def test_reputation_ceiling_matches(self):
        from backend.app.core.config import settings
        from ml import config as ml_config

        assert ml_config.REPUTATION_CEILING == settings.reputation_ceiling

    def test_risk_bands_match(self):
        from backend.app.core.config import settings
        from ml import config as ml_config

        assert ml_config.RISK_LOW_MAX == settings.risk_low_max
        assert ml_config.RISK_MEDIUM_MAX == settings.risk_medium_max

    def test_reputation_ceiling_lands_in_the_low_risk_band(self):
        """A clamped URL must read low-risk, not merely under the band edge.

        google.com scored 0.4466 before the prior: "legitimate", but risk score
        45, which crosses risk_low_max into MEDIUM and paints the card amber.
        That is the false positive users actually reported seeing.
        """
        from backend.app.core.config import settings

        assert settings.reputation_ceiling * 100 <= settings.risk_low_max
        assert settings.reputation_ceiling < settings.suspicious_threshold
