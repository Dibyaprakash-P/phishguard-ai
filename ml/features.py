"""Static URL feature engineering.

This module is the **single source of truth** for feature extraction. It is
used by the training pipeline (``ml/train.py``) and by the FastAPI inference
service (``backend/app/services/feature_extractor.py``) so that the feature
vector seen at training time is byte-for-byte identical to the one seen at
serving time.

Security note
-------------
Every feature here is derived from the URL **string only**. Nothing in this
module opens a socket, performs DNS resolution, issues an HTTP request or
executes anything from the submitted URL. This is a deliberate design
decision: it makes the analyzer immune to SSRF and to drive-by execution of
hostile content.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import parse_qsl, urlsplit

# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------

#: Tokens that frequently appear in credential-harvesting URLs. Derived from
#: common phishing lure vocabulary; used as a *signal*, never as a verdict.
SUSPICIOUS_KEYWORDS: tuple[str, ...] = (
    "login", "log-in", "signin", "sign-in", "verify", "verification", "account",
    "update", "secure", "security", "webscr", "banking", "confirm", "password",
    "credential", "wallet", "recover", "unlock", "suspend", "billing", "invoice",
    "payment", "customer", "support", "authenticate", "authorize", "session",
    "validate", "limited", "unusual", "alert", "urgent", "refund", "bonus",
    "gift", "prize", "winner", "claim", "activate", "reactivate", "myaccount",
)

#: Brand names most frequently impersonated. Presence of a brand *outside* the
#: registrable domain (e.g. in a subdomain or path) is a classic lure pattern.
TARGETED_BRANDS: tuple[str, ...] = (
    "paypal", "apple", "microsoft", "office365", "outlook", "google", "gmail",
    "amazon", "netflix", "facebook", "instagram", "whatsapp", "linkedin",
    "dropbox", "docusign", "adobe", "chase", "wellsfargo", "bankofamerica",
    "citibank", "hsbc", "barclays", "santander", "coinbase", "binance",
    "metamask", "blockchain", "steam", "roblox", "ebay", "alibaba", "dhl",
    "fedex", "ups", "usps", "irs", "hmrc", "icloud", "yahoo", "twitter",
)

#: Known URL-shortening services. Shorteners hide the true destination.
SHORTENER_DOMAINS: tuple[str, ...] = (
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "rebrand.ly", "shorte.st", "rb.gy",
    "tiny.cc", "lnkd.in", "db.tt", "qr.ae", "j.mp", "shorturl.at", "s.id",
    "clck.ru", "v.gd", "soo.gd", "u.to", "cli.gs", "trib.al", "t.ly",
)

#: TLDs disproportionately represented in abuse feeds (free / cheap registration).
SUSPICIOUS_TLDS: tuple[str, ...] = (
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click", "link",
    "loan", "download", "review", "country", "stream", "gdn", "racing",
    "win", "bid", "date", "faith", "cricket", "science", "party", "accountant",
    "zip", "mov", "rest", "buzz", "surf", "cyou", "icu", "monster", "quest",
)

#: Multi-label public suffixes needed to derive a registrable domain without a
#: network-backed Public Suffix List lookup. Covers the long tail seen in the
#: bundled datasets; anything else falls back to the last label.
_MULTI_PART_SUFFIXES: frozenset[str] = frozenset(
    f"{sld}.{cc}"
    for cc in (
        "uk", "au", "nz", "za", "br", "in", "jp", "kr", "cn", "tw", "hk", "sg",
        "id", "my", "th", "ph", "vn", "pk", "bd", "lk", "np", "il", "tr", "ua",
        "ru", "pl", "gr", "pt", "es", "it", "ar", "mx", "co", "ve", "pe", "ec",
        "eg", "sa", "ae", "ng", "ke", "gh", "tz", "ug", "zm", "zw", "ma", "dz",
    )
    for sld in ("co", "com", "net", "org", "gov", "edu", "ac", "mil", "sch", "gob", "or", "ne", "go")
)

_IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_HEX_IP_RE = re.compile(r"^0x[0-9a-f]+$", re.IGNORECASE)
_WORD_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_HEX_ESCAPE_RE = re.compile(r"%[0-9a-fA-F]{2}")

_SPECIAL_CHARS = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_VOWELS = frozenset("aeiou")

#: Ordered feature names. The order is contractual: the trained model consumes
#: the vector positionally, so **never reorder**; only append.
FEATURE_NAMES: tuple[str, ...] = (
    # --- coarse size features -------------------------------------------
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "fragment_length",
    "tld_length",
    # --- character counts -----------------------------------------------
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_digits",
    "num_letters",
    "num_special_chars",
    "num_equals",
    "num_ampersands",
    "num_question_marks",
    "num_percent",
    "num_at",
    "num_hash",
    "num_tilde",
    "num_plus",
    "num_commas",
    "num_semicolons",
    # --- ratios ----------------------------------------------------------
    "digit_ratio",
    "letter_ratio",
    "special_char_ratio",
    "host_digit_ratio",
    "host_vowel_ratio",
    # --- structure -------------------------------------------------------
    "num_subdomains",
    "path_depth",
    "num_query_params",
    "num_hyphens_in_host",
    "num_digits_in_host",
    "longest_host_token",
    "avg_host_token_length",
    "longest_path_token",
    "max_char_repeat",
    # --- entropy ---------------------------------------------------------
    "url_entropy",
    "domain_entropy",
    # --- binary indicators ------------------------------------------------
    "has_ip_host",
    "has_port",
    "has_at_symbol",
    "has_double_slash_in_path",
    "has_punycode",
    "has_hex_escape",
    "is_shortener",
    "has_suspicious_tld",
    "has_tld_in_subdomain",
    "has_tld_in_path",
    # --- lexical lures -----------------------------------------------------
    "num_suspicious_keywords",
    "brand_outside_domain",
    "num_brand_mentions",
)

FEATURE_COUNT: int = len(FEATURE_NAMES)

_COMMON_TLD_TOKENS = frozenset(
    ("com", "net", "org", "info", "biz", "co", "io", "gov", "edu", "ru", "de", "uk")
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _shannon_entropy(text: str) -> float:
    """Return the Shannon entropy (bits/char) of ``text``.

    Randomly generated / DGA-style domains carry noticeably higher entropy
    than human-readable ones.
    """
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _max_consecutive_repeat(text: str) -> int:
    """Length of the longest run of one repeated character."""
    best = current = 0
    previous = ""
    for char in text:
        current = current + 1 if char == previous else 1
        previous = char
        if current > best:
            best = current
    return best


def canonicalize_for_features(url: str) -> str:
    """Strip provenance markers before feature extraction.

    The bundled corpora record URLs in source-specific *styles* rather than a
    single canonical form. Measured over the assembled corpus:

    ==================  =====  =====  ========
    source (label 0)     www   https  has path
    ==================  =====  =====  ========
    PhiUSIIL            100%   100%   0%
    malicious_phish     0.04%  0.5%   100%
    ==================  =====  =====  ========

    Those columns describe *how each dataset was collected*, not whether a URL
    is a phishing page. A model trained on them learns to recognise the source
    corpus: it will classify ``https://www.x.com`` as legitimate and the
    equally benign ``https://google.com`` as phishing, purely because the
    latter's shape appears in neither legitimate style.

    Canonicalising the scheme and a leading ``www.`` away collapses the two
    legitimate styles onto each other, so the model must rely on the features
    that actually carry phishing semantics - hostname composition, lure
    vocabulary, entropy, TLD, structure - rather than on collection artefacts.

    The scheme is still reported to the user for transparency; it is simply not
    a model input. (In practice HTTPS is a weak signal anyway: certificates are
    free and the large majority of live phishing pages serve TLS.)
    """
    cleaned = (url or "").strip().strip('"').strip("'")
    if not cleaned:
        return ""
    cleaned = _SCHEME_RE.sub("", cleaned, count=1)
    if cleaned.lower().startswith("www."):
        cleaned = cleaned[4:]
    return cleaned


def normalize_url(url: str) -> str:
    """Normalise a raw URL string for parsing.

    Many public phishing corpora store URLs without a scheme
    (``www.example.com/login``). ``urlsplit`` would treat those as a bare path,
    so a scheme is prepended when missing. The original string is otherwise
    left untouched.
    """
    cleaned = (url or "").strip().strip('"').strip("'")
    if not cleaned:
        return ""
    if not _SCHEME_RE.match(cleaned):
        cleaned = "http://" + cleaned
    return cleaned


def _is_ip_host(host: str) -> bool:
    """True when the host component is a literal IP address."""
    if not host:
        return False
    candidate = host.strip("[]")
    if _IPV4_RE.match(candidate) or _HEX_IP_RE.match(candidate):
        try:
            if _HEX_IP_RE.match(candidate):
                return True
            ipaddress.ip_address(candidate)
            return True
        except ValueError:
            return False
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        return False


def registrable_domain(host: str) -> str:
    """Best-effort registrable domain ("example.co.uk") for ``host``.

    Uses a bundled multi-part suffix table rather than a live Public Suffix
    List lookup so that the function stays offline and deterministic. Used for
    group-aware dataset splitting, which is how domain-level leakage between
    train and test is prevented.
    """
    host = (host or "").lower().strip(".")
    if not host or _is_ip_host(host):
        return host
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if ".".join(labels[-2:]) in _MULTI_PART_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def split_url(url: str) -> dict[str, str]:
    """Split a URL into its components without touching the network."""
    normalized = normalize_url(url)
    try:
        parts = urlsplit(normalized)
        host = (parts.hostname or "").lower()
        return {
            "normalized": normalized,
            "scheme": (parts.scheme or "").lower(),
            "host": host,
            "port": str(parts.port) if parts.port else "",
            "path": parts.path or "",
            "query": parts.query or "",
            "fragment": parts.fragment or "",
            "netloc": parts.netloc or "",
        }
    except ValueError:
        # Malformed netloc (e.g. invalid IPv6 literal or bad port).
        return {
            "normalized": normalized,
            "scheme": "",
            "host": "",
            "port": "",
            "path": "",
            "query": "",
            "fragment": "",
            "netloc": "",
        }


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def extract_features(url: str) -> dict[str, float]:
    """Extract the full static feature dictionary for ``url``.

    Returns a mapping keyed by :data:`FEATURE_NAMES`. The function never
    raises for malformed input: unparseable URLs degrade to zero-valued
    structural features rather than aborting a batch.
    """
    # Provenance markers are stripped first - see canonicalize_for_features.
    parts = split_url(canonicalize_for_features(url))
    normalized = parts["normalized"]
    host = parts["host"]
    path = parts["path"]
    query = parts["query"]
    fragment = parts["fragment"]

    lowered = normalized.lower()
    host_labels = [label for label in host.split(".") if label]
    reg_domain = registrable_domain(host)
    tld = host_labels[-1] if host_labels else ""

    # Subdomain depth: labels in front of the registrable domain, ignoring www.
    reg_label_count = len(reg_domain.split(".")) if reg_domain else 0
    subdomain_labels = host_labels[: max(len(host_labels) - reg_label_count, 0)]
    effective_subdomains = len([s for s in subdomain_labels if s != "www"])

    url_len = len(normalized)
    num_digits = sum(c.isdigit() for c in normalized)
    num_letters = sum(c.isalpha() for c in normalized)
    num_special = sum(c in _SPECIAL_CHARS for c in normalized)

    host_digits = sum(c.isdigit() for c in host)
    host_alpha = sum(c.isalpha() for c in host)
    host_vowels = sum(c in _VOWELS for c in host)

    host_tokens = [t for t in _WORD_SPLIT_RE.split(host) if t]
    path_tokens = [t for t in _WORD_SPLIT_RE.split(path) if t]

    is_ip = _is_ip_host(host)
    keyword_hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in lowered)
    brand_hits = sum(1 for brand in TARGETED_BRANDS if brand in lowered)
    # A brand referenced anywhere except inside the registrable domain is the
    # classic "paypal.secure-login.example.com" impersonation pattern.
    outside_domain = lowered.replace(reg_domain, " ", 1) if reg_domain else lowered
    brand_outside = any(brand in outside_domain for brand in TARGETED_BRANDS)

    tld_in_subdomain = any(tok in _COMMON_TLD_TOKENS for tok in subdomain_labels)
    tld_in_path = any(tok in _COMMON_TLD_TOKENS for tok in path_tokens)

    features: dict[str, float] = {
        "url_length": float(url_len),
        "hostname_length": float(len(host)),
        "path_length": float(len(path)),
        "query_length": float(len(query)),
        "fragment_length": float(len(fragment)),
        "tld_length": float(len(tld)),

        "num_dots": float(normalized.count(".")),
        "num_hyphens": float(normalized.count("-")),
        "num_underscores": float(normalized.count("_")),
        "num_slashes": float(normalized.count("/")),
        "num_digits": float(num_digits),
        "num_letters": float(num_letters),
        "num_special_chars": float(num_special),
        "num_equals": float(normalized.count("=")),
        "num_ampersands": float(normalized.count("&")),
        "num_question_marks": float(normalized.count("?")),
        "num_percent": float(normalized.count("%")),
        "num_at": float(normalized.count("@")),
        "num_hash": float(normalized.count("#")),
        "num_tilde": float(normalized.count("~")),
        "num_plus": float(normalized.count("+")),
        "num_commas": float(normalized.count(",")),
        "num_semicolons": float(normalized.count(";")),

        "digit_ratio": num_digits / url_len if url_len else 0.0,
        "letter_ratio": num_letters / url_len if url_len else 0.0,
        "special_char_ratio": num_special / url_len if url_len else 0.0,
        "host_digit_ratio": host_digits / len(host) if host else 0.0,
        "host_vowel_ratio": host_vowels / host_alpha if host_alpha else 0.0,

        "num_subdomains": float(effective_subdomains),
        "path_depth": float(len([p for p in path.split("/") if p])),
        "num_query_params": float(len(parse_qsl(query, keep_blank_values=True))),
        "num_hyphens_in_host": float(host.count("-")),
        "num_digits_in_host": float(host_digits),
        "longest_host_token": float(max((len(t) for t in host_tokens), default=0)),
        "avg_host_token_length": (
            sum(len(t) for t in host_tokens) / len(host_tokens) if host_tokens else 0.0
        ),
        "longest_path_token": float(max((len(t) for t in path_tokens), default=0)),
        "max_char_repeat": float(_max_consecutive_repeat(normalized)),

        "url_entropy": _shannon_entropy(normalized),
        "domain_entropy": _shannon_entropy(host),

        "has_ip_host": 1.0 if is_ip else 0.0,
        "has_port": 1.0 if parts["port"] else 0.0,
        "has_at_symbol": 1.0 if "@" in normalized else 0.0,
        "has_double_slash_in_path": 1.0 if "//" in path else 0.0,
        "has_punycode": 1.0 if "xn--" in host else 0.0,
        "has_hex_escape": 1.0 if _HEX_ESCAPE_RE.search(normalized) else 0.0,
        "is_shortener": 1.0 if reg_domain in SHORTENER_DOMAINS else 0.0,
        "has_suspicious_tld": 1.0 if tld in SUSPICIOUS_TLDS else 0.0,
        "has_tld_in_subdomain": 1.0 if tld_in_subdomain else 0.0,
        "has_tld_in_path": 1.0 if tld_in_path else 0.0,

        "num_suspicious_keywords": float(keyword_hits),
        "brand_outside_domain": 1.0 if brand_outside else 0.0,
        "num_brand_mentions": float(brand_hits),
    }

    # Guard against silent contract drift between FEATURE_NAMES and the dict.
    assert len(features) == FEATURE_COUNT, "feature dict/FEATURE_NAMES mismatch"
    return features


def extract_feature_vector(url: str) -> list[float]:
    """Extract features as a positionally ordered vector."""
    features = extract_features(url)
    return [features[name] for name in FEATURE_NAMES]


def matched_suspicious_keywords(url: str) -> list[str]:
    """Return which suspicious keywords actually occur in ``url``.

    Used by the API so the UI and the LLM prompt can cite concrete evidence
    instead of an opaque count.
    """
    lowered = normalize_url(canonicalize_for_features(url)).lower()
    return [kw for kw in SUSPICIOUS_KEYWORDS if kw in lowered]


def matched_brands(url: str) -> list[str]:
    """Return which impersonation-target brand names occur in ``url``."""
    lowered = normalize_url(canonicalize_for_features(url)).lower()
    return [brand for brand in TARGETED_BRANDS if brand in lowered]
