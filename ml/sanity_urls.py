"""Curated smoke-test URLs used as a post-training quality gate.

Why this exists
---------------
An earlier build of this project reached 92% test accuracy while confidently
classifying ``https://google.com`` as phishing. The cause was a
provenance shortcut in the corpora (see
:func:`ml.features.canonicalize_for_features`): every legitimate URL in one
source was ``https://www.<domain>`` with no path, and every legitimate URL in
another had a path and no scheme, so a bare ``https://google.com`` matched
neither legitimate style.

Aggregate test metrics could not surface that, because the artifact is present
in the test split too. A small curated set of obviously-correct cases can, and
does. The training pipeline runs this set after selecting a model and fails
loudly when accuracy on it is poor.

The four groups, and why each is separate
-----------------------------------------
:data:`LEGITIMATE_URLS`
    Well-known sites, most of which are on the curated known-good list in
    :mod:`ml.reputation`. The reputation prior makes these easy *as served*,
    which is the point - it is what the user sees - but it also means their
    score says little about the model.

:data:`UNLISTED_LEGITIMATE_URLS`
    Real legitimate sites deliberately **absent** from the known-good list.
    Nothing shields these, so they are the honest read on whether the model
    itself recognises benign structure. A reputation list must never be able
    to paper over a model regression, and this group is what stops it.

:data:`PHISHING_SHAPED_URLS`
    Textbook lure structure over reserved/example domains.

:data:`BYPASS_URLS`
    Attacks specifically shaped to borrow a trusted name: brand tokens outside
    the registrable domain, user-content hosts under a reputable domain, and
    the ``@``-in-authority trick. Every one of these must stay flagged; if the
    reputation prior ever admits one, it has become a vulnerability.

Honesty note
------------
This set is hand-written by the author and is deliberately easy. It is a smoke
test, **not** a benchmark: its score is reported separately from the held-out
test metrics and must never be quoted as model accuracy.
"""

from __future__ import annotations

#: Well-known legitimate URLs, deliberately spanning the stylistic axes that
#: the corpora confound: bare vs. www, http vs. https, homepage vs. deep path,
#: short vs. long, with and without query strings.
LEGITIMATE_URLS: tuple[str, ...] = (
    "https://google.com",
    "https://www.google.com",
    "google.com",
    "https://github.com",
    "https://github.com/torvalds/linux",
    "https://www.wikipedia.org",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://stackoverflow.com/questions/tagged/python",
    "https://www.amazon.com",
    "https://www.microsoft.com",
    "https://openai.com",
    "https://www.bbc.co.uk/news",
    "https://news.ycombinator.com",
    "https://arxiv.org/abs/1706.03762",
    "https://www.python.org/downloads/",
    "https://pypi.org/project/fastapi/",
    "https://docs.python.org/3/library/urllib.parse.html",
    "https://www.nytimes.com",
    "https://mail.google.com",
    "https://www.linkedin.com/in/example",
    "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://azure.microsoft.com/en-us/products/ai-services/openai-service",
    "https://scikit-learn.org/stable/modules/ensemble.html",
    "https://www.gov.uk/browse/benefits",
    # Added after measuring the shipped artifact: every one of these was
    # reported as suspicious or phishing. paypal.com, dropbox.com,
    # instagram.com and netflix.com are the starkest cases - a brand token in
    # the URL is scored as an impersonation signal even when the URL *is* the
    # brand's own apex domain.
    "https://paypal.com",
    "https://dropbox.com",
    "https://instagram.com",
    "https://netflix.com",
    "https://linkedin.com",
    "https://cloudflare.com",
    "https://apple.com",
    "https://adobe.com",
    "https://spotify.com",
    "https://chatgpt.com",
    "https://facebook.com",
    "https://x.com",
    "https://reddit.com",
)

#: Legitimate URLs deliberately **not** on the known-good list in
#: :mod:`ml.reputation`. These receive no reputation help whatsoever, so they
#: are the group that actually measures the classifier. Keep them off that
#: list: the moment one is added, it stops testing anything.
UNLISTED_LEGITIMATE_URLS: tuple[str, ...] = (
    # --- developer documentation -----------------------------------------
    "https://www.sqlite.org/lang_select.html",
    "https://www.postgresql.org/docs/current/sql-select.html",
    "https://nginx.org/en/docs/http/ngx_http_core_module.html",
    "https://curl.se/docs/manpage.html",
    "https://www.openssl.org/docs/man3.0/man1/openssl.html",
    "https://ffmpeg.org/ffmpeg-filters.html",
    "https://redis.io/docs/latest/commands/set/",
    "https://www.rfc-editor.org/rfc/rfc7231",
    "https://man7.org/linux/man-pages/man1/ls.1.html",
    "https://mariadb.org/documentation/",
    "https://www.mysql.com/products/community/",
    "https://jquery.com/download/",
    "https://lodash.com/docs/",
    "https://d3js.org/getting-started",
    "https://leafletjs.com/examples.html",
    "https://www.chartjs.org/docs/latest/",
    "https://getbootstrap.com/docs/5.3/getting-started/introduction/",
    "https://sass-lang.com/documentation/",
    "https://caniuse.com/flexbox",
    "https://regex101.com/",
    # --- reference and media ----------------------------------------------
    "https://www.gutenberg.org/ebooks/1342",
    "https://openlibrary.org/search?q=phishing",
    "https://www.imdb.com/title/tt0111161/",
    "https://www.britannica.com/topic/phishing",
    "https://www.merriam-webster.com/dictionary/phishing",
    "https://dictionary.cambridge.org/dictionary/english/phishing",
    "https://www.goodreads.com/book/show/1.Harry_Potter",
    "https://www.rottentomatoes.com/m/inception",
    "https://www.espn.com/nfl/scoreboard",
    "https://www.allrecipes.com/recipe/213742/",
    "https://www.smithsonianmag.com/science-nature/",
    "https://www.nationalgeographic.com/animals/",
    # --- services, public bodies, education -------------------------------
    "https://www.weather.gov/forecast",
    "https://www.timeanddate.com/worldclock/",
    "https://www.nhs.uk/conditions/",
    "https://www.who.int/health-topics/",
    "https://www.worldbank.org/en/research",
    "https://www.coursera.org/learn/machine-learning",
    "https://www.khanacademy.org/math/algebra",
    "https://www.indeed.com/jobs?q=engineer",
)

#: URLs exhibiting textbook phishing structure. These are constructed patterns
#: over reserved/example domains, not live sites - the point is to check that
#: clear lure structure is still detected after canonicalisation.
PHISHING_SHAPED_URLS: tuple[str, ...] = (
    "http://192.168.1.10/login/verify.php",
    "http://paypal.com.secure-login-verify.tk/webscr?cmd=_account-update",
    "https://secure-paypa1-login.example.com/verify/account/confirm",
    "http://appleid-apple.com-verify-account.gq/signin/unlock",
    "http://update-your-account-now.microsoft.security-alert.ml/login.php",
    "http://bankofamerica.com.session-expired-confirm.cf/update/billing",
    "http://netflix-billing-update-required.xyz/account/payment/verify",
    "http://amaz0n-security-alert.top/signin?verify=urgent&account=suspend",
    "http://dhl-parcel-redelivery-payment.icu/track/confirm/billing",
    "http://coinbase-wallet-recover-seed.buzz/unlock/validate",
    "http://192.3.44.201:8080/wp-content/paypal/webscr/login/confirm.php",
    "http://steamcommunity-tradeoffer-verify.gq/login/authenticate",
    "http://irs-refund-claim-2024.work/validate/ssn/confirm",
    "http://office365-outlook-password-expired.click/session/reauthenticate",
    "http://wellsfargo.com-online-banking-alert.date/verify/customer/login",
)

#: Attacks shaped to borrow a known-good name. Each targets one specific rule
#: in :mod:`ml.reputation`; a regression here means the reputation prior has
#: become a bypass rather than a convenience.
BYPASS_URLS: tuple[str, ...] = (
    # Brand as a subdomain label of an attacker-owned registrable domain.
    "http://paypal.com.secure-login-verify.tk/webscr?cmd=_account-update",
    # Brand token inside an attacker-registered domain.
    "https://paypal-support.com/verify/account",
    "http://microsoft.com-verify-account.gq/signin/unlock",
    # User content under a reputable registrable domain.
    "https://evil.github.io/login/verify",
    "https://sites.google.com/view/verify-your-account/home",
    "https://myfiles.blob.core.windows.net/share/office365-login.html",
    # Authority confusion: everything before the @ is userinfo, not the host.
    "http://google.com@evil.tk/signin",
    # Homograph-style lookalike.
    "https://secure-paypa1-login.example.com/verify/account/confirm",
)


def sanity_cases() -> list[tuple[str, int]]:
    """Return ``[(url, label)]`` with 0 = legitimate, 1 = phishing."""
    return (
        [(url, 0) for url in LEGITIMATE_URLS]
        + [(url, 0) for url in UNLISTED_LEGITIMATE_URLS]
        + [(url, 1) for url in PHISHING_SHAPED_URLS]
        + [(url, 1) for url in BYPASS_URLS]
    )


def sanity_groups() -> dict[str, tuple[tuple[str, ...], int]]:
    """Return ``{group_name: (urls, label)}`` for per-group reporting.

    The groups are scored separately because they answer different questions -
    see the module docstring. Collapsing them into one number is what let the
    original ``google.com`` bug survive a 92%-accurate model.
    """
    return {
        "well_known": (LEGITIMATE_URLS, 0),
        "unlisted_legitimate": (UNLISTED_LEGITIMATE_URLS, 0),
        "phishing_shaped": (PHISHING_SHAPED_URLS, 1),
        "bypass_attempts": (BYPASS_URLS, 1),
    }
