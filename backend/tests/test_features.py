"""Tests for the shared static feature extractor."""

from __future__ import annotations

import math

import pytest

from ml.features import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    canonicalize_for_features,
    extract_feature_vector,
    extract_features,
    matched_brands,
    matched_suspicious_keywords,
    normalize_url,
    registrable_domain,
    split_url,
)


class TestContract:
    def test_feature_names_are_unique(self):
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)

    def test_extract_returns_every_declared_feature(self):
        features = extract_features("https://example.com/path?a=1")
        assert set(features) == set(FEATURE_NAMES)
        assert len(features) == FEATURE_COUNT

    def test_vector_order_matches_feature_names(self):
        url = "https://secure-login.example.com/verify"
        features = extract_features(url)
        vector = extract_feature_vector(url)
        assert vector == [features[name] for name in FEATURE_NAMES]

    def test_all_values_are_finite_numbers(self):
        for url in ("https://a.com", "http://1.2.3.4:8080/x?y=z#frag", "a.b.c.d.e.f.gg"):
            for name, value in extract_features(url).items():
                assert isinstance(value, float), name
                assert math.isfinite(value), name

    def test_provenance_features_are_not_model_inputs(self):
        # These encode which corpus a URL came from, not whether it is phishing.
        assert "has_https" not in FEATURE_NAMES
        assert "has_www_prefix" not in FEATURE_NAMES


class TestCanonicalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("https://www.google.com", "google.com"),
            ("http://google.com", "google.com"),
            ("google.com", "google.com"),
            ("HTTPS://WWW.Google.com", "WWW.Google.com".replace("WWW.", "")),
        ],
    )
    def test_scheme_and_www_are_stripped(self, raw, expected):
        assert canonicalize_for_features(raw) == expected

    def test_equivalent_urls_produce_identical_features(self):
        """The core defence against the corpus provenance shortcut."""
        assert extract_features("https://www.google.com") == extract_features("http://google.com")

    def test_only_leading_www_is_stripped(self):
        # www inside the host is meaningful (e.g. a lure subdomain) and stays.
        assert canonicalize_for_features("http://paypal.com.www.evil.tk") == (
            "paypal.com.www.evil.tk"
        )


class TestRobustness:
    @pytest.mark.parametrize("value", ["", "   ", "http://", "://", "http://[bad", "!!!"])
    def test_malformed_input_never_raises(self, value):
        features = extract_features(value)
        assert len(features) == FEATURE_COUNT

    def test_empty_url_yields_zeroed_features(self):
        features = extract_features("")
        assert features["url_length"] == 0.0
        assert features["domain_entropy"] == 0.0

    def test_very_long_url_is_handled(self):
        features = extract_features("http://example.com/" + "a" * 5000)
        assert features["url_length"] > 5000


class TestIndividualFeatures:
    def test_ip_host_detected(self):
        assert extract_features("http://192.168.1.10/login")["has_ip_host"] == 1.0
        assert extract_features("http://example.com/login")["has_ip_host"] == 0.0

    def test_at_symbol_detected(self):
        assert extract_features("http://evil.com@real.com/x")["has_at_symbol"] == 1.0

    def test_punycode_detected(self):
        assert extract_features("http://xn--80ak6aa92e.com")["has_punycode"] == 1.0

    def test_shortener_detected(self):
        assert extract_features("https://bit.ly/abc")["is_shortener"] == 1.0
        assert extract_features("https://example.com/abc")["is_shortener"] == 0.0

    def test_suspicious_tld_detected(self):
        assert extract_features("http://free-stuff.tk")["has_suspicious_tld"] == 1.0
        assert extract_features("http://example.com")["has_suspicious_tld"] == 0.0

    def test_subdomain_count_excludes_www(self):
        assert extract_features("http://www.example.com")["num_subdomains"] == 0.0
        assert extract_features("http://a.b.example.com")["num_subdomains"] == 2.0

    def test_port_detected(self):
        assert extract_features("http://example.com:8080/x")["has_port"] == 1.0

    def test_entropy_is_higher_for_random_hostnames(self):
        readable = extract_features("http://google.com")["domain_entropy"]
        random_like = extract_features("http://x7k2qp9zvw3mnb.com")["domain_entropy"]
        assert random_like > readable

    def test_digit_ratio(self):
        features = extract_features("http://a1234.com")
        assert 0 < features["digit_ratio"] < 1

    def test_brand_outside_registrable_domain(self):
        impersonation = extract_features("http://paypal.com.secure-login.evil.tk/verify")
        genuine = extract_features("http://paypal.com/signin")
        assert impersonation["brand_outside_domain"] == 1.0
        assert genuine["brand_outside_domain"] == 0.0

    def test_suspicious_keyword_count(self):
        features = extract_features("http://x.com/login/verify/account")
        assert features["num_suspicious_keywords"] >= 3


class TestHelpers:
    @pytest.mark.parametrize(
        "host,expected",
        [
            ("www.google.com", "google.com"),
            ("google.com", "google.com"),
            ("a.b.example.co.uk", "example.co.uk"),
            ("example.com.br", "example.com.br"),
            ("192.168.0.1", "192.168.0.1"),
        ],
    )
    def test_registrable_domain(self, host, expected):
        assert registrable_domain(host) == expected

    def test_normalize_adds_missing_scheme(self):
        assert normalize_url("example.com/x").startswith("http://")
        assert normalize_url("https://example.com") == "https://example.com"

    def test_split_url_components(self):
        parts = split_url("https://a.example.com:8443/p/q?x=1#f")
        assert parts["scheme"] == "https"
        assert parts["host"] == "a.example.com"
        assert parts["port"] == "8443"
        assert parts["path"] == "/p/q"
        assert parts["query"] == "x=1"
        assert parts["fragment"] == "f"

    def test_matched_keywords_returns_evidence(self):
        matched = matched_suspicious_keywords("http://x.com/secure/login")
        assert "login" in matched and "secure" in matched

    def test_matched_brands_returns_evidence(self):
        assert "paypal" in matched_brands("http://paypal-verify.tk")


class TestBrandMentions:
    """`num_brand_mentions` must signal impersonation, not brand presence.

    Two defects were measured on the shipped artifact and are fixed here:

    * a brand token inside the brand's **own** registrable domain was counted,
      so `paypal.com` scored 0.6532 and `dropbox.com` 0.7971 - above the
      phishing threshold - for being themselves;
    * matching was a plain substring scan, so "ups" fired on
      `groups.google.com` and "chase" on `purchase`.
    """

    @pytest.mark.parametrize("url", [
        "https://paypal.com",
        "https://paypal.com/signin",
        "https://dropbox.com",
        "https://microsoft.com",
        "https://netflix.com",
        "https://instagram.com",
        "https://google.com",
        "https://mail.google.com",
        "https://groups.google.com",
        "https://icloud.com",
        "https://x.com",
    ])
    def test_a_brand_on_its_own_domain_is_not_a_mention(self, url):
        assert extract_features(url)["num_brand_mentions"] == 0.0
        assert matched_brands(url) == []

    @pytest.mark.parametrize("url,brand", [
        ("http://paypal.com.secure-login-verify.tk/webscr", "paypal"),
        ("http://evil.tk/paypal/login", "paypal"),
        ("https://paypal-support.com/verify/account", "paypal"),
        ("http://appleid-apple.com-verify-account.gq/signin", "apple"),
        ("http://netflix-billing-update-required.xyz/account", "netflix"),
        ("http://update-now.microsoft.security-alert.ml/login.php", "microsoft"),
    ])
    def test_impersonation_still_counts(self, url, brand):
        assert extract_features(url)["num_brand_mentions"] >= 1.0
        assert brand in matched_brands(url)

    @pytest.mark.parametrize("url", [
        "https://purchase-tickets.example.com/order",   # "chase" inside "purchase"
        "https://startups.example.com/list",            # "ups" inside "startups"
        "https://backups.example.org/nightly",          # "ups" inside "backups"
    ])
    def test_short_brands_do_not_match_inside_ordinary_words(self, url):
        assert extract_features(url)["num_brand_mentions"] == 0.0

    def test_long_brands_still_match_as_an_affix(self):
        # No separator, so a token-equality-only rule would miss these.
        assert "paypal" in matched_brands("https://paypalsecure.example.tk/login")
        assert "microsoft" in matched_brands("https://securemicrosoft.example.tk/x")

    def test_brand_on_its_own_domain_still_counts_when_repeated_elsewhere(self):
        # The exemption covers the registrable domain only. A brand that also
        # appears in the path is a real signal and must survive.
        features = extract_features("https://paypal.com/paypal-verify/login")
        assert features["num_brand_mentions"] >= 1.0

    def test_lookalike_domain_is_not_exempt(self):
        # paypa1 (digit one) is not paypal and is not on any exemption list.
        assert extract_features("https://paypa1.com/verify")["num_brand_mentions"] == 0.0
        # ...but the brand token in a hyphenated lookalike is caught.
        assert "paypal" in matched_brands("https://paypal-secure.com/verify")

    def test_every_brand_domain_key_is_a_targeted_brand(self):
        from ml.features import BRAND_DOMAINS, TARGETED_BRANDS

        assert set(BRAND_DOMAINS) <= set(TARGETED_BRANDS)

    def test_every_brand_domain_value_is_a_registrable_domain(self):
        from ml.features import BRAND_DOMAINS, registrable_domain

        for brand, domains in BRAND_DOMAINS.items():
            for domain in domains:
                assert registrable_domain(domain) == domain, f"{brand}: {domain}"
