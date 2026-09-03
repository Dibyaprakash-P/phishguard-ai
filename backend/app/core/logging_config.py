"""Logging setup with privacy-preserving URL redaction.

PhishGuard is stateless and database-free. Submitted URLs are never persisted;
when a URL must appear in a log line for debugging, it is redacted first so
that credentials, tokens and session identifiers carried in the query string
are not written to disk.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from backend.app.core.config import settings

#: Query parameters whose values are replaced wholesale.
SENSITIVE_PARAMS = frozenset({
    "token", "access_token", "id_token", "refresh_token", "auth", "authorization",
    "key", "api_key", "apikey", "secret", "client_secret", "password", "passwd",
    "pwd", "pass", "session", "sessionid", "sid", "otp", "code", "email", "user",
    "username", "login", "account", "card", "cvv", "ssn", "phone",
})

_REDACTED = "[REDACTED]"


def redact_url(url: str, max_length: int = 120) -> str:
    """Return a log-safe representation of ``url``.

    Scheme, host and path shape are preserved because they are what makes a log
    line useful; sensitive query values are stripped and the result is
    truncated. A short hash is appended so repeated occurrences of the same URL
    can still be correlated without storing the URL itself.
    """
    if not url:
        return ""
    digest = hashlib.sha256(url.encode("utf-8", "replace")).hexdigest()[:10]
    try:
        parts = urlsplit(url if "://" in url else f"http://{url}")
        pairs = parse_qsl(parts.query, keep_blank_values=True)
        safe_query = urlencode(
            [(k, _REDACTED if k.lower() in SENSITIVE_PARAMS else v) for k, v in pairs]
        )
        safe = urlunsplit((parts.scheme, parts.netloc, parts.path, safe_query, ""))
    except ValueError:
        safe = "<unparseable>"
    if len(safe) > max_length:
        safe = safe[:max_length] + "..."
    return f"{safe} (sha256:{digest})"


def configure_logging() -> None:
    """Install a single stdout handler at the configured level."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)

    # Uvicorn installs its own noisy handlers; route them through ours.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers = [handler]
        logging.getLogger(name).propagate = False
