"""End-to-end API tests against a stubbed model (no artifact, no API key)."""

from __future__ import annotations

import pytest


class TestHealth:
    def test_health_reports_loaded_model(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert "llm" in body

    def test_health_reports_degraded_without_model(self, offline_client):
        body = offline_client.get("/api/health").json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False

    def test_root_lists_endpoints(self, client):
        body = client.get("/").json()
        assert "/api/analyze" in body["endpoints"]
        assert body["static_analysis_only"] is True


class TestAnalyze:
    def test_returns_full_schema(self, client):
        body = client.post("/api/analyze", json={"url": "https://example.com"}).json()
        for field in (
            "url", "normalized_url", "prediction", "risk_level", "confidence",
            "phishing_probability", "risk_score", "components", "features",
            "feature_highlights", "suspicious_indicators", "ai_explanation",
            "model_name", "decision_threshold", "analysis_ms",
        ):
            assert field in body, f"missing {field}"

    def test_static_analysis_flag_is_always_true(self, client):
        body = client.post("/api/analyze", json={"url": "https://example.com"}).json()
        assert body["static_analysis_only"] is True

    def test_risk_score_matches_probability(self, client):
        body = client.post("/api/analyze", json={"url": "http://192.168.1.10/login"}).json()
        assert body["risk_score"] == round(body["phishing_probability"] * 100)

    def test_indicators_are_returned_for_a_suspicious_url(self, client):
        body = client.post(
            "/api/analyze", json={"url": "http://paypal.com.verify-login.tk/account"}
        ).json()
        codes = {i["code"] for i in body["suspicious_indicators"]}
        assert "brand_impersonation" in codes
        assert "suspicious_keywords" in codes

    @pytest.mark.parametrize(
        "url", ["", "   ", "not a url", "javascript:alert(1)", "ftp://example.com", "localhost"]
    )
    def test_invalid_urls_return_422_with_message(self, client, url):
        response = client.post("/api/analyze", json={"url": url})
        assert response.status_code == 422
        body = response.json()
        assert body["error"] in {"invalid_url", "validation_error"}
        assert body["message"]

    def test_missing_url_field_returns_422(self, client):
        assert client.post("/api/analyze", json={}).status_code == 422

    def test_no_stack_trace_leaks_to_the_client(self, client):
        body = client.post("/api/analyze", json={"url": "javascript:alert(1)"}).json()
        serialised = str(body)
        assert "Traceback" not in serialised
        assert "File \"" not in serialised

    def test_returns_503_when_model_missing(self, offline_client):
        response = offline_client.post("/api/analyze", json={"url": "https://example.com"})
        assert response.status_code == 503
        assert response.json()["error"] == "model_unavailable"


class TestLLMFallback:
    """Without credentials the API must still answer, clearly labelled."""

    def test_fallback_explanation_is_served_and_labelled(self, client):
        explanation = client.post(
            "/api/analyze", json={"url": "https://example.com"}
        ).json()["ai_explanation"]

        assert explanation["source"] == "rule_based"
        assert explanation["available"] is False
        assert "unavailable" in explanation["notice"].lower()
        assert len(explanation["text"]) > 80

    def test_fallback_never_claims_the_site_was_visited(self, client):
        text = client.post(
            "/api/analyze", json={"url": "http://paypal-verify.tk/login"}
        ).json()["ai_explanation"]["text"].lower()

        # Affirmative claims about checks that were never performed. The text
        # is allowed - and expected - to *disclaim* these, so the assertions
        # target the claim, not the mere presence of the word.
        for forbidden in (
            "we visited", "we fetched", "the page contains", "malware was found",
            "whois shows", "whois lookup revealed", "the certificate is",
            "blocklist match", "was downloaded",
        ):
            assert forbidden not in text

    def test_fallback_states_that_no_external_data_was_consulted(self, client):
        text = client.post(
            "/api/analyze", json={"url": "http://paypal-verify.tk/login"}
        ).json()["ai_explanation"]["text"].lower()
        assert "not visited" in text

    def test_explanation_can_be_declined(self, client):
        explanation = client.post(
            "/api/analyze",
            json={"url": "https://example.com", "include_ai_explanation": False},
        ).json()["ai_explanation"]
        assert explanation["source"] == "rule_based"


class TestBatch:
    def test_batch_returns_one_row_per_url(self, client):
        body = client.post(
            "/api/batch-analyze",
            json={"urls": ["https://a.com", "https://b.com", "http://192.168.1.1/login"]},
        ).json()
        assert len(body["results"]) == 3
        assert body["analyzed"] == 3
        assert body["failed"] == 0

    def test_invalid_rows_are_isolated(self, client):
        body = client.post(
            "/api/batch-analyze", json={"urls": ["https://a.com", "not a url"]}
        ).json()
        assert body["analyzed"] == 1
        assert body["failed"] == 1
        assert body["results"][0]["prediction"] is not None
        assert body["results"][1]["error"]

    def test_batch_limit_enforced(self, client):
        response = client.post(
            "/api/batch-analyze", json={"urls": [f"https://x{i}.com" for i in range(200)]}
        )
        assert response.status_code == 422

    def test_empty_batch_rejected(self, client):
        assert client.post("/api/batch-analyze", json={"urls": []}).status_code == 422


class TestModelInfo:
    def test_reports_untrained_state_without_fake_metrics(self, offline_client):
        body = offline_client.get("/api/model-info").json()
        assert body["trained"] is False
        assert body["test_metrics"] is None
        assert body["message"]

    def test_reports_metadata_when_loaded(self, client):
        body = client.get("/api/model-info").json()
        assert body["trained"] is True
        assert body["feature_count"] > 0
        assert body["feature_names"]


class TestAgent:
    def test_agent_disabled_by_default(self, client):
        response = client.post("/api/agent-analyze", json={"url": "https://example.com"})
        assert response.status_code == 501
        assert response.json()["error"] == "agent_unavailable"
