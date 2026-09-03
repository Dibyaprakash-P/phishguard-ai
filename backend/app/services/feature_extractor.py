"""URL validation and presentation layer over the shared feature extractor.

The numeric feature vector itself comes from :mod:`ml.features` - the exact
module used during training - so there is no possibility of train/serve skew.
What this module adds is:

* strict input validation with actionable error messages,
* human-readable feature cards for the dashboard,
* a list of **observed** suspicious indicators, each carrying the literal
  evidence found in the URL string.

Nothing here performs DNS resolution, HTTP requests or any other network I/O.
"""

from __future__ import annotations

import ipaddress
import logging
from urllib.parse import urlsplit

from backend.app.core.config import settings
from backend.app.core.exceptions import InvalidURLError
from backend.app.schemas.analysis import (
    FeatureHighlight,
    Severity,
    SuspiciousIndicator,
    URLComponents,
)
from ml.features import (
    FEATURE_NAMES,
    extract_features,
    matched_brands,
    matched_suspicious_keywords,
    normalize_url,
    registrable_domain,
)

logger = logging.getLogger(__name__)

#: Schemes we are willing to analyze. Anything else (javascript:, data:,
#: file:, ftp:) is rejected outright rather than silently coerced.
ALLOWED_SCHEMES = frozenset({"http", "https"})

#: Schemes that are actively dangerous to echo back into a browser context.
BLOCKED_SCHEMES = frozenset({
    "javascript", "data", "vbscript", "file", "about", "blob", "chrome", "jar",
})


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_url(raw: str) -> tuple[str, URLComponents]:
    """Validate ``raw`` and return ``(normalized_url, components)``.

    Raises :class:`InvalidURLError` with a specific message for every rejection
    reason so the frontend can show something better than "invalid input".
    """
    candidate = (raw or "").strip()
    if not candidate:
        raise InvalidURLError("Please enter a URL to analyze.")

    if len(candidate) > settings.max_url_length:
        raise InvalidURLError(
            f"URL is too long ({len(candidate)} characters). "
            f"The limit is {settings.max_url_length}."
        )

    if any(ch in candidate for ch in ("\n", "\r", "\t", " ")):
        raise InvalidURLError("URL must not contain whitespace or line breaks.")

    # Reject dangerous schemes before normalisation would hide them.
    scheme_head = candidate.split(":", 1)[0].lower() if ":" in candidate else ""
    if scheme_head in BLOCKED_SCHEMES:
        raise InvalidURLError(
            f"'{scheme_head}:' URLs cannot be analyzed. Provide an http or https URL."
        )

    normalized = normalize_url(candidate)
    try:
        parts = urlsplit(normalized)
        host = (parts.hostname or "").lower()
        port = str(parts.port) if parts.port else None
    except ValueError as exc:
        raise InvalidURLError("The URL could not be parsed.", str(exc)) from exc

    if parts.scheme not in ALLOWED_SCHEMES:
        raise InvalidURLError(
            f"Unsupported scheme '{parts.scheme}'. Only http and https URLs are analyzed."
        )

    if not host:
        raise InvalidURLError("The URL is missing a hostname.")

    if len(host) > 253:
        raise InvalidURLError("The hostname exceeds the maximum DNS length of 253 characters.")

    # A host must be either an IP literal or a dotted name. This is a
    # structural check on the string; no DNS lookup is performed.
    if not _is_ip_literal(host):
        if "." not in host:
            raise InvalidURLError(
                f"'{host}' is not a valid public hostname. Include a domain suffix, "
                "for example example.com."
            )
        if host.startswith(".") or host.endswith(".") or ".." in host:
            raise InvalidURLError("The hostname contains empty domain labels.")
        if any(len(label) > 63 for label in host.split(".")):
            raise InvalidURLError("A domain label exceeds the maximum length of 63 characters.")

    domain = registrable_domain(host)
    labels = [lbl for lbl in host.split(".") if lbl]
    components = URLComponents(
        scheme=parts.scheme,
        host=host,
        registrable_domain=domain,
        port=port,
        path=parts.path or "/",
        query=parts.query or "",
        fragment=parts.fragment or "",
        tld=labels[-1] if labels and not _is_ip_literal(host) else "",
    )
    return normalized, components


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

def extract(url: str) -> dict[str, float]:
    """Return the canonical feature dictionary for ``url``."""
    return extract_features(url)


def feature_vector(features: dict[str, float]) -> list[float]:
    """Order a feature dict into the model's positional input vector."""
    return [features[name] for name in FEATURE_NAMES]


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------

def _tone(value: float, medium: float, high: float) -> Severity:
    if value >= high:
        return Severity.HIGH
    if value >= medium:
        return Severity.MEDIUM
    return Severity.INFO


def build_feature_highlights(
    features: dict[str, float], components: URLComponents
) -> list[FeatureHighlight]:
    """Select the features worth showing as cards in the dashboard."""
    keywords = int(features["num_suspicious_keywords"])
    is_https = components.scheme == "https"
    highlights = [
        FeatureHighlight(
            key="url_length", label="URL Length",
            value=f"{int(features['url_length'])}",
            raw_value=features["url_length"], icon="Ruler",
            description="Total characters. Unusually long URLs can hide the real destination.",
            tone=_tone(features["url_length"], 75, 120),
        ),
        FeatureHighlight(
            key="hostname_length", label="Domain Length",
            value=f"{int(features['hostname_length'])}",
            raw_value=features["hostname_length"], icon="Globe",
            description="Characters in the hostname.",
            tone=_tone(features["hostname_length"], 30, 45),
        ),
        FeatureHighlight(
            key="num_subdomains", label="Subdomains",
            value=f"{int(features['num_subdomains'])}",
            raw_value=features["num_subdomains"], icon="Network",
            description="Labels before the registrable domain, excluding www.",
            tone=_tone(features["num_subdomains"], 2, 4),
        ),
        FeatureHighlight(
            key="num_special_chars", label="Special Characters",
            value=f"{int(features['num_special_chars'])}",
            raw_value=features["num_special_chars"], icon="Hash",
            description="Punctuation and symbols across the whole URL.",
            tone=_tone(features["num_special_chars"], 18, 30),
        ),
        FeatureHighlight(
            key="num_digits", label="Digits",
            value=f"{int(features['num_digits'])}",
            raw_value=features["num_digits"], icon="Binary",
            description="Numeric characters. Dense digits often indicate generated hosts.",
            tone=_tone(features["num_digits"], 10, 20),
        ),
        FeatureHighlight(
            key="has_https", label="HTTPS",
            value="Enabled" if is_https else "Not used",
            raw_value=1.0 if is_https else 0.0, icon="Lock",
            description=(
                "Transport encryption. Shown for transparency but deliberately NOT a "
                "model input - most live phishing sites serve valid TLS."
            ),
            tone=Severity.INFO if is_https else Severity.LOW,
        ),
        FeatureHighlight(
            key="has_ip_host", label="IP Address Host",
            value="Yes" if features["has_ip_host"] else "No",
            raw_value=features["has_ip_host"], icon="Server",
            description="A raw IP in place of a domain name bypasses brand recognition.",
            tone=Severity.HIGH if features["has_ip_host"] else Severity.INFO,
        ),
        FeatureHighlight(
            key="num_suspicious_keywords", label="Suspicious Keywords",
            value=f"{keywords}",
            raw_value=features["num_suspicious_keywords"], icon="KeyRound",
            description="Lure vocabulary such as login, verify, secure or account.",
            tone=_tone(keywords, 2, 4),
        ),
        FeatureHighlight(
            key="domain_entropy", label="Domain Entropy",
            value=f"{features['domain_entropy']:.2f}",
            raw_value=features["domain_entropy"], icon="Activity",
            description="Shannon entropy in bits/char. High values suggest random hostnames.",
            tone=_tone(features["domain_entropy"], 3.6, 4.2),
        ),
        FeatureHighlight(
            key="path_depth", label="Path Depth",
            value=f"{int(features['path_depth'])}",
            raw_value=features["path_depth"], icon="FolderTree",
            description="Number of path segments after the domain.",
            tone=_tone(features["path_depth"], 4, 7),
        ),
        FeatureHighlight(
            key="num_query_params", label="Query Parameters",
            value=f"{int(features['num_query_params'])}",
            raw_value=features["num_query_params"], icon="ListFilter",
            description="Key/value pairs in the query string.",
            tone=_tone(features["num_query_params"], 4, 8),
        ),
        FeatureHighlight(
            key="digit_ratio", label="Digit Ratio",
            value=f"{features['digit_ratio'] * 100:.1f}%",
            raw_value=features["digit_ratio"], icon="Percent",
            description="Share of the URL made up of digits.",
            tone=_tone(features["digit_ratio"], 0.15, 0.3),
        ),
    ]
    return highlights


def build_suspicious_indicators(
    url: str, features: dict[str, float], components: URLComponents
) -> list[SuspiciousIndicator]:
    """Enumerate the concrete, observable red flags present in the URL string.

    Each entry records the literal evidence that triggered it. These are
    factual observations about the text - they are deliberately kept separate
    from the model's probabilistic verdict.
    """
    indicators: list[SuspiciousIndicator] = []

    def add(code: str, title: str, description: str,
            severity: Severity, evidence: str | None = None) -> None:
        indicators.append(SuspiciousIndicator(
            code=code, title=title, description=description,
            severity=severity, evidence=evidence,
        ))

    if features["has_ip_host"]:
        add("ip_host", "IP address used as hostname",
            "The URL points at a raw IP address instead of a registered domain name, "
            "which prevents the user from recognising the brand they expect.",
            Severity.HIGH, components.host)

    if features["has_at_symbol"]:
        add("at_symbol", "'@' symbol in URL",
            "Everything before an '@' in the authority section is treated as user "
            "info and ignored by browsers, so the real host can be disguised.",
            Severity.HIGH, "@")

    if features["has_punycode"]:
        add("punycode", "Punycode hostname",
            "The hostname uses an 'xn--' encoded label, which can render as "
            "characters visually identical to a well-known brand.",
            Severity.HIGH, components.host)

    if features["is_shortener"]:
        add("shortener", "URL shortening service",
            "The link is served by a shortening service, so the final destination "
            "is not visible in the URL itself.",
            Severity.MEDIUM, components.registrable_domain)

    brands = matched_brands(url)
    if features["brand_outside_domain"] and brands:
        add("brand_impersonation", "Brand name outside the registered domain",
            f"The name(s) {', '.join(brands[:3])} appear in the URL but not as the "
            f"registrable domain, which is '{components.registrable_domain}'. This is a "
            "common impersonation pattern.",
            Severity.HIGH, ", ".join(brands[:3]))

    keywords = matched_suspicious_keywords(url)
    if keywords:
        add("suspicious_keywords", "Credential-harvesting vocabulary",
            f"The URL contains {len(keywords)} term(s) frequently used in phishing lures.",
            Severity.HIGH if len(keywords) >= 3 else Severity.MEDIUM,
            ", ".join(keywords[:6]))

    if features["has_suspicious_tld"]:
        add("suspicious_tld", "High-abuse top-level domain",
            f"'.{components.tld}' is over-represented in abuse feeds, largely because "
            "registration is free or very cheap.",
            Severity.MEDIUM, f".{components.tld}")

    if components.scheme != "https":
        add("no_https", "No HTTPS",
            "The URL is not served over TLS, so submitted data would travel in the "
            "clear. This is an observation only - it is not a model input, because "
            "HTTPS is free and most live phishing sites use it.",
            Severity.LOW, components.scheme)

    if features["has_port"]:
        add("non_standard_port", "Explicit port number",
            "The URL specifies a port explicitly, which is unusual for consumer-facing "
            "websites.",
            Severity.LOW, components.port)

    if features["num_subdomains"] >= 3:
        add("deep_subdomains", "Deeply nested subdomains",
            f"The hostname has {int(features['num_subdomains'])} subdomain levels. Long "
            "chains are used to push the real domain out of view on mobile browsers.",
            Severity.MEDIUM, components.host)

    if features["url_length"] >= 100:
        add("long_url", "Unusually long URL",
            f"At {int(features['url_length'])} characters this URL is far longer than a "
            "typical link, which helps hide its true target.",
            Severity.MEDIUM, f"{int(features['url_length'])} characters")

    if features["has_double_slash_in_path"]:
        add("double_slash", "Double slash in path",
            "A '//' inside the path is often the residue of an embedded redirect target.",
            Severity.LOW, "//")

    if features["has_hex_escape"]:
        add("percent_encoding", "Percent-encoded characters",
            "The URL contains %XX escape sequences, which can obscure readable text.",
            Severity.LOW, "%")

    if features["num_hyphens_in_host"] >= 3:
        add("hyphenated_host", "Heavily hyphenated hostname",
            f"The hostname contains {int(features['num_hyphens_in_host'])} hyphens, a "
            "pattern used to imitate brand names such as 'secure-login-brand'.",
            Severity.MEDIUM, components.host)

    if features["domain_entropy"] >= 4.0:
        add("high_entropy_domain", "High-entropy hostname",
            f"Character entropy of {features['domain_entropy']:.2f} bits/char suggests an "
            "algorithmically generated rather than a human-chosen name.",
            Severity.MEDIUM, components.host)

    if features["has_tld_in_subdomain"]:
        add("tld_in_subdomain", "Domain suffix inside a subdomain",
            "A suffix such as '.com' appears in a subdomain label, making the URL look "
            "like a familiar domain while resolving elsewhere.",
            Severity.MEDIUM, components.host)

    if features["host_digit_ratio"] >= 0.3:
        add("numeric_host", "Digit-heavy hostname",
            f"{features['host_digit_ratio'] * 100:.0f}% of the hostname is digits.",
            Severity.LOW, components.host)

    severity_rank = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2, Severity.INFO: 3}
    return sorted(indicators, key=lambda i: severity_rank[i.severity])
