"""Tests for the LLM explanation service and its deterministic fallback.

No test here makes a network call. The LLM chain is mocked, which is also what
lets CI run without any API key.
"""

from __future__ import annotations

import asyncio

import pytest

from backend.app.schemas.analysis import (
    ExplanationSource,
    Prediction,
    Severity,
    SuspiciousIndicator,
)
from backend.app.services.feature_extractor import extract, validate_url
from backend.app.services.llm_analyzer import (
    HUMAN_PROMPT,
    SYSTEM_PROMPT,
    LLMAnalyzer,
    build_rule_based_explanation,
)


@pytest.fixture
def sample():
    url = "http://paypal.com.secure-login.evil.tk/verify/account"
    normalized, components = validate_url(url)
    features = extract(normalized)
    indicators = [
        SuspiciousIndicator(
            code="brand_impersonation", title="Brand name outside the registered domain",
            description="Test indicator.", severity=Severity.HIGH, evidence="paypal",
        ),
        SuspiciousIndicator(
            code="suspicious_tld", title="High-abuse top-level domain",
            description="Test indicator.", severity=Severity.MEDIUM, evidence=".tk",
        ),
    ]
    return normalized, components, features, indicators


class TestSystemPrompt:
    """The prompt is the primary guard against fabricated evidence."""

    @pytest.mark.parametrize(
        "constraint",
        ["NOT visited", "WHOIS", "malware", "Use ONLY the evidence provided",
         "statistical prediction", "blocklist"],
    )
    def test_prompt_forbids_fabrication(self, constraint):
        assert constraint in SYSTEM_PROMPT

    def test_prompt_receives_the_final_verdict(self):
        assert "already final" in HUMAN_PROMPT
        assert "{prediction}" in HUMAN_PROMPT
        assert "{indicators}" in HUMAN_PROMPT


class TestRuleBasedFallback:
    def test_produces_substantive_text(self, sample):
        _, components, features, indicators = sample
        text = build_rule_based_explanation(
            Prediction.PHISHING, 0.96, 96, components, features, indicators
        )
        assert len(text) > 150
        assert "\n\n" in text  # multiple paragraphs

    def test_cites_only_supplied_indicators(self, sample):
        _, components, features, indicators = sample
        text = build_rule_based_explanation(
            Prediction.PHISHING, 0.96, 96, components, features, indicators
        ).lower()
        assert "brand name outside the registered domain" in text

    def test_never_claims_unperformed_checks(self, sample):
        _, components, features, indicators = sample
        for prediction, probability in (
            (Prediction.PHISHING, 0.96), (Prediction.SUSPICIOUS, 0.55),
            (Prediction.LEGITIMATE, 0.04),
        ):
            text = build_rule_based_explanation(
                prediction, probability, int(probability * 100), components, features, indicators
            ).lower()
            for forbidden in ("whois", "we visited", "page content", "downloaded", "malware"):
                assert forbidden not in text

    def test_states_that_the_site_was_not_visited(self, sample):
        _, components, features, indicators = sample
        text = build_rule_based_explanation(
            Prediction.LEGITIMATE, 0.05, 5, components, features, []
        )
        assert "not visited" in text.lower()

    def test_handles_zero_indicators(self, sample):
        _, components, features, _ = sample
        text = build_rule_based_explanation(
            Prediction.LEGITIMATE, 0.02, 2, components, features, []
        )
        assert "no red-flag patterns" in text.lower()


class TestExplain:
    """Explanation routing.

    ``asyncio.run`` is used instead of pytest-asyncio so the suite needs no
    extra plugin, which keeps the CI install minimal.
    """

    @staticmethod
    def _explain(analyzer, sample, **kwargs):
        _, components, features, indicators = sample
        return asyncio.run(
            analyzer.explain(
                prediction=Prediction.PHISHING, probability=0.96, risk_score=96,
                risk_level="high", components=components, features=features,
                indicators=indicators, model_name="test", model_version="1.0",
                threshold=0.5, **kwargs,
            )
        )

    def test_falls_back_when_no_provider_configured(self, sample):
        result = self._explain(LLMAnalyzer(), sample)
        assert result.source is ExplanationSource.RULE_BASED
        assert result.available is False
        assert "unavailable" in result.notice.lower()

    def test_uses_llm_output_when_the_chain_succeeds(self, sample):
        class FakeChain:
            async def ainvoke(self, _variables):
                return "  A concise security explanation from the model.  "

        analyzer = LLMAnalyzer()
        analyzer._initialized = True
        analyzer._chain = FakeChain()
        analyzer._model_label = "openai:test-model"

        result = self._explain(analyzer, sample)
        assert result.source is ExplanationSource.LLM
        assert result.available is True
        assert result.text == "A concise security explanation from the model."
        assert result.model == "openai:test-model"
        assert result.notice is None

    def test_provider_failure_degrades_to_fallback(self, sample):
        class BrokenChain:
            async def ainvoke(self, _variables):
                raise TimeoutError("provider timed out")

        analyzer = LLMAnalyzer()
        analyzer._initialized = True
        analyzer._chain = BrokenChain()
        analyzer._model_label = "openai:test-model"

        result = self._explain(analyzer, sample)
        assert result.source is ExplanationSource.RULE_BASED
        assert result.available is False
        assert "TimeoutError" in result.notice
        # The ML-derived analysis is still delivered, not an error page.
        assert len(result.text) > 150

    def test_empty_llm_response_degrades_to_fallback(self, sample):
        class EmptyChain:
            async def ainvoke(self, _variables):
                return "   "

        analyzer = LLMAnalyzer()
        analyzer._initialized = True
        analyzer._chain = EmptyChain()

        assert self._explain(analyzer, sample).source is ExplanationSource.RULE_BASED

    def test_llm_can_be_declined_per_request(self, sample):
        class FakeChain:
            async def ainvoke(self, _variables):
                raise AssertionError("the chain must not be invoked")

        analyzer = LLMAnalyzer()
        analyzer._initialized = True
        analyzer._chain = FakeChain()

        assert self._explain(analyzer, sample, want_llm=False).source is (
            ExplanationSource.RULE_BASED
        )

    def test_prompt_carries_only_the_supplied_evidence(self, sample):
        """The model must not be handed anything it could mistake for a new fact."""
        captured = {}

        class CapturingChain:
            async def ainvoke(self, variables):
                captured.update(variables)
                return "ok"

        analyzer = LLMAnalyzer()
        analyzer._initialized = True
        analyzer._chain = CapturingChain()

        self._explain(analyzer, sample)

        assert captured["prediction"] == "phishing"
        assert "evil.tk" in captured["host"]
        # Exactly the two indicators supplied by the fixture, no more.
        assert captured["indicator_count"] == 2
        assert captured["measurements"].count("\n") >= 5
