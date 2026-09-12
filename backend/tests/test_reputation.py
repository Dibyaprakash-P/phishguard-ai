"""Tests for the curated known-good domain reputation prior.

The prior lowers the score for URLs on a hand-curated list of registrable
domains, because a URL-only classifier cannot read a bare domain and so parks
well-known sites near its prior. That is a convenience for the user and a
liability if it is sloppy: anything that lets an attacker borrow a trusted name
turns the list into a bypass. Most of what follows tests the refusals.
"""

from __future__ import annotations

import pytest

from ml.features import registrable_domain
from ml.reputation import (
    KNOWN_GOOD_DOMAINS,
    USER_CONTENT_DOMAINS,
    USER_CONTENT_HOSTS,
    is_known_good,
    known_good_domain,
)


class TestListHygiene:
    """Invariants on the reference data itself.

    An entry that is not its own registrable domain is dead weight - the lookup
    compares against ``registrable_domain(host)``, so ``aws.amazon.com`` in the
    list would never match anything while *looking* like it grants trust.
    """

    @pytest.mark.parametrize("domain", sorted(KNOWN_GOOD_DOMAINS))
    def test_known_good_entry_is_its_own_registrable_domain(self, domain):
        assert registrable_domain(domain) == domain

    @pytest.mark.parametrize("domain", sorted(USER_CONTENT_DOMAINS))
    def test_user_content_entry_is_its_own_registrable_domain(self, domain):
        assert registrable_domain(domain) == domain

    def test_the_two_domain_sets_are_disjoint(self):
        # An overlap would make trust depend on which set is consulted first.
        assert not (KNOWN_GOOD_DOMAINS & USER_CONTENT_DOMAINS)

    def test_user_content_hosts_are_hosts_not_registrable_domains(self):
        # These are listed as full hosts precisely because their registrable
        # domain is trusted and must stay trusted.
        for host in USER_CONTENT_HOSTS:
            assert registrable_domain(host) != host


class TestMatches:
    @pytest.mark.parametrize("url,expected", [
        ("https://google.com", "google.com"),
        ("https://www.google.com", "google.com"),
        ("google.com", "google.com"),
        ("http://google.com", "google.com"),
        ("https://github.com/torvalds/linux", "github.com"),
        ("https://mail.google.com", "google.com"),
        ("https://www.bbc.co.uk/news", "bbc.co.uk"),
        ("https://paypal.com", "paypal.com"),
        ("https://azure.microsoft.com/en-us/products", "microsoft.com"),
    ])
    def test_known_good_urls_match(self, url, expected):
        assert known_good_domain(url) == expected

    def test_scheme_and_www_do_not_change_the_answer(self):
        assert len({
            known_good_domain("github.com"),
            known_good_domain("http://github.com"),
            known_good_domain("https://www.github.com/"),
        }) == 1


class TestRefusals:
    """Every case here is an attack that must not inherit trust."""

    @pytest.mark.parametrize("url,why", [
        ("http://paypal.com.secure-login-verify.tk/webscr",
         "brand is a subdomain label; registrable domain is secure-login-verify.tk"),
        ("http://microsoft.com-verify-account.gq/signin",
         "brand token glued to an attacker-owned domain"),
        ("https://paypal-support.com/verify",
         "contains a brand token but is not the brand's domain"),
        ("https://secure-paypa1-login.example.com/verify",
         "homograph-style lookalike"),
        ("https://evil.github.io/login",
         "user content under a reputable registrable domain"),
        ("https://sites.google.com/view/verify-account",
         "user-content host under a trusted registrable domain"),
        ("https://myfiles.blob.core.windows.net/share/login.html",
         "object storage under windows.net"),
        ("http://google.com@evil.tk/signin",
         "everything before the @ is userinfo, not the host"),
        ("https://xn--goog-sla.com",
         "punycode host - what is rendered is not what resolves"),
        ("http://192.168.1.10/login/verify.php", "IP literal host"),
        ("https://notion.site/anything", "user-publishable domain"),
        ("https://bit.ly/3xyz", "shortener hides the destination"),
        ("https://not-a-real-domain-xyzzy.test/", "simply not on the list"),
        ("", "empty input"),
    ])
    def test_refused(self, url, why):
        assert known_good_domain(url) is None, why

    def test_trust_does_not_flow_from_a_listed_domain_to_its_user_content_host(self):
        # google.com is trusted; sites.google.com must not be.
        assert known_good_domain("https://google.com") == "google.com"
        assert known_good_domain("https://sites.google.com/view/x") is None

    def test_brand_in_path_does_not_match(self):
        assert known_good_domain("http://evil.tk/paypal.com/login") is None

    def test_is_known_good_agrees_with_known_good_domain(self):
        for url in ("https://github.com", "https://evil.github.io/x", "http://1.2.3.4/"):
            assert is_known_good(url) == (known_good_domain(url) is not None)


class TestReputationPrior:
    """The clamp applied at serving time."""

    def test_clamps_a_high_scoring_known_good_domain(self):
        from backend.app.services.predictor import apply_reputation

        probability, domain = apply_reputation(0.80, "https://dropbox.com")
        assert domain == "dropbox.com"
        assert probability == pytest.approx(0.25)

    def test_never_raises_a_score(self):
        from backend.app.services.predictor import apply_reputation

        # Already below the ceiling - reputation must leave it alone rather
        # than pulling it up to 0.25.
        probability, domain = apply_reputation(0.01, "https://wikipedia.org")
        assert domain == "wikipedia.org"
        assert probability == pytest.approx(0.01)

    def test_leaves_unlisted_domains_untouched(self):
        from backend.app.services.predictor import apply_reputation

        probability, domain = apply_reputation(0.93, "http://evil-login-verify.tk/x")
        assert domain is None
        assert probability == pytest.approx(0.93)

    def test_can_be_disabled(self, monkeypatch):
        from backend.app.core.config import settings
        from backend.app.services.predictor import apply_reputation

        monkeypatch.setattr(settings, "reputation_ceiling", 1.0)
        probability, domain = apply_reputation(0.80, "https://dropbox.com")
        assert domain is None
        assert probability == pytest.approx(0.80)
