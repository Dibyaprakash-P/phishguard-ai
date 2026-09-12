"""Curated known-good domain reputation.

Why this exists
---------------
A URL-only classifier reads lexical structure, and a bare, path-less domain
carries almost none. Measured against the shipped artifact, 15 of 52
hand-picked well-known legitimate URLs were not reported as legitimate:
``dropbox.com`` scored 0.797 and ``paypal.com`` 0.653 (both above the phishing
threshold), while ``microsoft.com``, ``linkedin.com`` and ``mail.google.com``
landed in the suspicious band at 0.58-0.62. ``google.com`` (0.447) and
``github.com`` (0.407) sat just under it - close enough to the model's prior
that the risk score still rendered as medium.

Two causes, neither fixable by moving a threshold: the training corpora
under-represent bare legitimate domains, and :data:`ml.features.TARGETED_BRANDS`
scores a brand token anywhere in the URL, so a brand's *own* apex domain is
charged for an impersonation pattern it is the victim of.

What this module is
-------------------
A reputation **prior**, answering one narrow question: is this URL served from
a domain independently established as legitimate? It is deliberately not a
classifier and not an allowlist of URLs - only the registrable domain is
consulted, and only an exact match counts.

Security constraints
--------------------
Each rule below exists because omitting it creates a bypass:

* **Exact registrable-domain match only.** ``paypal.com.secure-login.tk`` has
  registrable domain ``secure-login.tk`` and never matches. Neither does
  ``paypal-support.com``, which merely contains a brand token.
* **User-content hosts are excluded.** ``github.io``, ``sites.google.com`` and
  ``blogspot.com`` serve attacker-controlled pages under a registrable domain
  that is otherwise trustworthy, and are among the most abused hosts in real
  phishing feeds. Trust must not flow to them.
* **Obfuscated hosts never match.** An ``@`` in the authority or a punycode
  label means the host a human reads is not the host a browser resolves, which
  is the whole point of the lure.

Residual risk, stated plainly
-----------------------------
A compromised page or open redirect on a genuinely legitimate domain is not
detectable from the URL string, so this module does not make the product blind
to anything it could otherwise have seen - that class of attack was already out
of scope for a URL-only model.

The list is hand-curated and deliberately small. It is not a substitute for the
model, it is not consulted for any URL off these domains, and every match is
surfaced to the user as an explicit indicator rather than applied silently.
"""

from __future__ import annotations

from ml.features import registrable_domain, split_url

# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------

#: Registrable domains established as legitimate. Two groups are represented:
#: the official domains of every brand in :data:`ml.features.TARGETED_BRANDS`
#: (which the brand-mention feature otherwise penalises for being themselves),
#: and high-traffic sites a user is likely to paste into a checker.
KNOWN_GOOD_DOMAINS: frozenset[str] = frozenset({
    # --- search, portals, infrastructure ---------------------------------
    "google.com", "google.co.uk", "google.de", "google.fr", "google.co.in",
    "bing.com", "duckduckgo.com", "yahoo.com", "yandex.com", "baidu.com",
    "cloudflare.com", "akamai.com", "fastly.com", "digitalocean.com",
    # --- big tech ---------------------------------------------------------
    "apple.com", "icloud.com", "microsoft.com", "live.com", "office.com",
    "office365.com", "outlook.com", "msn.com", "azure.com",
    "amazon.com", "amazon.co.uk", "amazon.de", "amazon.in",
    "meta.com", "facebook.com", "instagram.com", "whatsapp.com", "threads.net",
    "x.com", "twitter.com", "linkedin.com", "netflix.com", "spotify.com",
    "adobe.com", "oracle.com", "ibm.com", "intel.com", "nvidia.com",
    "salesforce.com", "sap.com", "vmware.com", "cisco.com", "dell.com",
    "hp.com", "lenovo.com", "samsung.com", "sony.com", "lg.com",
    # --- developer / technical -------------------------------------------
    "github.com", "gitlab.com", "bitbucket.org", "stackoverflow.com",
    "stackexchange.com", "python.org", "pypi.org", "npmjs.com", "nodejs.org",
    "rust-lang.org", "golang.org", "go.dev", "kernel.org", "gnu.org",
    "apache.org", "mozilla.org", "w3.org", "ietf.org", "docker.com",
    "kubernetes.io", "scikit-learn.org", "numpy.org", "pydata.org",
    "pytorch.org", "tensorflow.org", "huggingface.co", "kaggle.com",
    "jetbrains.com", "visualstudio.com", "atlassian.com", "slack.com",
    "zoom.us", "dropbox.com", "box.com", "notion.so", "figma.com",
    "postman.com", "cloudinary.com", "heroku.com", "vercel.com", "netlify.com",
    # --- AI ---------------------------------------------------------------
    "openai.com", "chatgpt.com", "anthropic.com", "claude.ai",
    "perplexity.ai", "mistral.ai", "cohere.com", "deepmind.com",
    # --- finance (the most impersonated category) -------------------------
    "paypal.com", "stripe.com", "squareup.com", "wise.com", "revolut.com",
    "chase.com", "wellsfargo.com", "bankofamerica.com", "citibank.com",
    "citi.com", "capitalone.com", "usbank.com", "pnc.com", "truist.com",
    "amex.com", "americanexpress.com", "discover.com", "visa.com",
    "mastercard.com", "hsbc.com", "hsbc.co.uk", "barclays.co.uk",
    "lloydsbank.com", "natwest.com", "santander.co.uk", "nationwide.co.uk",
    "monzo.com", "starlingbank.com", "schwab.com", "fidelity.com",
    "vanguard.com", "blackrock.com", "goldmansachs.com", "morganstanley.com",
    "hdfcbank.com", "icicibank.com", "sbi.co.in", "axisbank.com", "paytm.com",
    "coinbase.com", "binance.com", "kraken.com", "gemini.com", "ledger.com",
    # --- commerce / travel ------------------------------------------------
    "ebay.com", "etsy.com", "shopify.com", "alibaba.com", "aliexpress.com",
    "walmart.com", "target.com", "bestbuy.com", "costco.com", "ikea.com",
    "flipkart.com", "myntra.com", "booking.com", "airbnb.com", "expedia.com",
    "uber.com", "lyft.com", "doordash.com", "instacart.com",
    # --- shipping (heavily impersonated) ----------------------------------
    "dhl.com", "fedex.com", "ups.com", "usps.com", "royalmail.com",
    "canadapost.ca", "auspost.com.au",
    # --- media / reference ------------------------------------------------
    "wikipedia.org", "wikimedia.org", "bbc.co.uk", "bbc.com", "cnn.com",
    "nytimes.com", "washingtonpost.com", "theguardian.com", "reuters.com",
    "apnews.com", "bloomberg.com", "ft.com", "economist.com", "wsj.com",
    "npr.org", "aljazeera.com", "thehindu.com", "indiatimes.com",
    "youtube.com", "vimeo.com", "twitch.tv", "reddit.com", "quora.com",
    "ycombinator.com", "arxiv.org",
    "nature.com", "science.org", "sciencedirect.com", "springer.com",
    "ieee.org", "acm.org", "jstor.org", "doi.org", "orcid.org",
    # --- government / education ------------------------------------------
    "gov.uk", "usa.gov", "irs.gov", "nih.gov", "nasa.gov", "cdc.gov",
    "europa.eu", "hmrc.gov.uk", "india.gov.in", "canada.ca",
    "mit.edu", "stanford.edu", "harvard.edu", "berkeley.edu", "ox.ac.uk",
    "cam.ac.uk", "ethz.ch", "uci.edu", "cmu.edu", "caltech.edu",
    # --- other commonly pasted -------------------------------------------
    "docusign.com", "docusign.net", "mailchimp.com", "sendgrid.com",
    "zendesk.com", "hubspot.com", "asana.com", "trello.com", "monday.com",
    "steampowered.com", "roblox.com", "epicgames.com", "playstation.com",
    "xbox.com", "nintendo.com", "discord.com", "telegram.org", "signal.org",
})

#: Registrable domains that host third-party, attacker-controllable content.
#: These are **never** trusted even though the registrable domain itself is
#: reputable, because anyone can publish under them. They are listed
#: explicitly rather than merely omitted so that adding, say, ``google.com``
#: above can never silently confer trust on ``sites.google.com``.
USER_CONTENT_DOMAINS: frozenset[str] = frozenset({
    "github.io", "gitlab.io", "githubusercontent.com", "gitbook.io",
    "blogspot.com", "wordpress.com", "wixsite.com", "weebly.com",
    "squarespace.com", "webflow.io", "glitch.me", "repl.co", "replit.app",
    "netlify.app", "vercel.app", "web.app", "firebaseapp.com", "pages.dev",
    "herokuapp.com", "azurewebsites.net", "cloudfront.net", "amazonaws.com",
    "windows.net", "googleapis.com", "appspot.com", "googleusercontent.com",
    "000webhostapp.com", "duckdns.org", "ngrok.io", "ngrok-free.app",
    "sharepoint.com", "notion.site", "medium.com", "substack.com",
    "forms.gle", "typeform.com", "jotform.com", "surveymonkey.com",
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "linktr.ee",
})

#: Hosts under an otherwise-trusted registrable domain that serve user content.
#: Matched on the full host, because the registrable domain alone (``google.com``)
#: is legitimately trusted while these specific hosts are not.
USER_CONTENT_HOSTS: frozenset[str] = frozenset({
    "sites.google.com", "docs.google.com", "drive.google.com",
    "script.google.com", "groups.google.com", "forms.google.com",
    "colab.research.google.com",
    "onedrive.live.com", "sway.office.com", "forms.office.com",
})


# --------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------

def known_good_domain(url: str) -> str | None:
    """Return the matched registrable domain, or ``None`` when not known-good.

    A match requires *all* of the following, and any one of them failing
    returns ``None``:

    1. the host parses and is not an IP literal;
    2. the host is not a user-content host, and its registrable domain is not
       a user-content domain;
    3. the authority carries no ``@`` and no punycode label;
    4. the registrable domain is present in :data:`KNOWN_GOOD_DOMAINS` exactly.

    >>> known_good_domain("https://github.com/torvalds/linux")
    'github.com'
    >>> known_good_domain("http://paypal.com.secure-login-verify.tk/webscr")
    >>> known_good_domain("https://evil.github.io/login")
    """
    parts = split_url(url)
    host = parts["host"]
    if not host:
        return None

    # An @ anywhere in the authority means the text before it is not the host
    # the browser resolves - the classic "google.com@evil.tk" lure.
    if "@" in parts["netloc"] or "xn--" in host:
        return None

    if host in USER_CONTENT_HOSTS:
        return None

    domain = registrable_domain(host)
    if not domain or domain in USER_CONTENT_DOMAINS:
        return None

    return domain if domain in KNOWN_GOOD_DOMAINS else None


def is_known_good(url: str) -> bool:
    """True when :func:`known_good_domain` matches."""
    return known_good_domain(url) is not None
