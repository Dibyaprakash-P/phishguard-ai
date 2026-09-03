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


def sanity_cases() -> list[tuple[str, int]]:
    """Return ``[(url, label)]`` with 0 = legitimate, 1 = phishing."""
    return (
        [(url, 0) for url in LEGITIMATE_URLS]
        + [(url, 1) for url in PHISHING_SHAPED_URLS]
    )
