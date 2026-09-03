"""Tests for URL validation and the observed-indicator builder."""

from __future__ import annotations

import pytest

from backend.app.core.exceptions import InvalidURLError
from backend.app.schemas.analysis import Severity
from backend.app.services.feature_extractor import (
    build_feature_highlights,
    build_suspicious_indicators,
    extract,
    validate_url,
)


class TestValidateAccepts:
    @pytest.mark.parametrize(
        "url",
        [
            "https://google.com",
            "http://example.com",
            "example.com",
            "www.example.com/path",
            "https://sub.domain.example.co.uk/a/b?c=1#d",
            "http://192.168.1.10/login",
            "http://example.com:8080/x",
            # Suspicious-looking URLs must NOT be rejected: classifying them is
            # the entire purpose of the product.
            "http://paypal.com.secure-login-verify.tk/webscr",
            "https://secure-paypa1-login.example.com/verify",
        ],
    )
    def test_accepted(self, url):
        normalized, components = validate_url(url)
        assert normalized
        assert components.host


class TestValidateRejects:
    @pytest.mark.parametrize(
        "url,fragment",
        [
            ("", "enter a url"),
            ("   ", "enter a url"),
            ("not a url", "whitespace"),
            ("has space.com/x", "whitespace"),
            ("javascript:alert(1)", "cannot be analyzed"),
            ("data:text/html,<script>", "cannot be analyzed"),
            ("file:///etc/passwd", "cannot be analyzed"),
            ("ftp://example.com", "unsupported scheme"),
            ("localhost", "domain suffix"),
            ("http://.com", "empty domain labels"),
            ("http://a..b.com", "empty domain labels"),
        ],
    )
    def test_rejected_with_specific_message(self, url, fragment):
        with pytest.raises(InvalidURLError) as excinfo:
            validate_url(url)
        assert fragment in str(excinfo.value).lower()

    def test_rejects_over_length_url(self):
        with pytest.raises(InvalidURLError, match="too long"):
            validate_url("http://example.com/" + "a" * 5000)

    def test_rejects_oversized_domain_label(self):
        with pytest.raises(InvalidURLError, match="63 characters"):
            validate_url(f"http://{'a' * 70}.com")


class TestComponents:
    def test_components_are_parsed(self):
        _, components = validate_url("https://a.b.example.co.uk:8443/p?q=1#f")
        assert components.scheme == "https"
        assert components.host == "a.b.example.co.uk"
        assert components.registrable_domain == "example.co.uk"
        assert components.port == "8443"
        assert components.tld == "uk"

    def test_missing_scheme_defaults_to_http(self):
        normalized, components = validate_url("example.com")
        assert normalized == "http://example.com"
        assert components.scheme == "http"


class TestIndicators:
    def _indicators(self, url):
        normalized, components = validate_url(url)
        return build_suspicious_indicators(normalized, extract(normalized), components)

    def test_clean_url_yields_no_high_severity_indicators(self):
        codes = {i.code for i in self._indicators("https://github.com")}
        assert "ip_host" not in codes
        assert "brand_impersonation" not in codes

    def test_ip_host_flagged_high(self):
        indicator = next(i for i in self._indicators("http://192.168.1.10/x") if i.code == "ip_host")
        assert indicator.severity == Severity.HIGH

    def test_indicators_carry_evidence(self):
        for indicator in self._indicators("http://paypal.com.secure-login.evil.tk/verify/account"):
            assert indicator.evidence, f"{indicator.code} has no evidence"

    def test_brand_impersonation_detected(self):
        codes = {i.code for i in self._indicators("http://paypal.com.secure-login.evil.tk/verify")}
        assert "brand_impersonation" in codes

    def test_indicators_sorted_by_severity(self):
        indicators = self._indicators("http://192.168.1.10:8080/login/verify/account/confirm")
        rank = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2, Severity.INFO: 3}
        ranks = [rank[i.severity] for i in indicators]
        assert ranks == sorted(ranks)

    def test_https_is_reported_from_components_not_features(self):
        # has_https was removed as a model feature but must still be observable.
        codes = {i.code for i in self._indicators("http://example.com")}
        assert "no_https" in codes
        assert "no_https" not in {i.code for i in self._indicators("https://example.com")}


class TestHighlights:
    def test_highlights_cover_expected_metrics(self):
        normalized, components = validate_url("https://example.com/a/b?c=1")
        highlights = build_feature_highlights(extract(normalized), components)
        keys = {h.key for h in highlights}
        assert {"url_length", "has_https", "num_subdomains", "domain_entropy"} <= keys

    def test_https_highlight_reflects_the_real_scheme(self):
        for url, expected in (("https://example.com", "Enabled"), ("http://example.com", "Not used")):
            normalized, components = validate_url(url)
            highlights = build_feature_highlights(extract(normalized), components)
            assert next(h for h in highlights if h.key == "has_https").value == expected
