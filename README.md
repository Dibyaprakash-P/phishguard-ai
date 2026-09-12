# PhishGuard AI

**Phishing URL detection and AI security analysis.**

A machine-learning classifier scores URLs from statically extracted lexical and
structural features; an LLM then explains that verdict through LangChain. The
model decides — the LLM only describes what it decided.

The analyzer **never visits the URL you give it**. Every feature is computed
from the URL string, which is what makes the service immune to SSRF and to
executing hostile content. There is no database anywhere in the system.

---

## Table of contents

- [Highlights](#highlights)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Dataset](#dataset)
- [Feature engineering](#feature-engineering)
- [Machine learning](#machine-learning)
- [Results](#results)
- [The bug that shaped this project](#the-bug-that-shaped-this-project)
- [LLM integration (LangChain)](#llm-integration-langchain)
- [Agentic AI (LangGraph)](#agentic-ai-langgraph)
- [MLflow](#mlflow)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Docker](#docker)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Security considerations](#security-considerations)
- [Privacy](#privacy)
- [Azure deployment path](#azure-deployment-path)
- [Limitations](#limitations)
- [Future improvements](#future-improvements)
- [Resume bullet points](#resume-bullet-points)
- [Interview preparation](#interview-preparation)

---

## Highlights

- **Real ML, honestly measured.** Five models compared; the winner is selected
  on validation PR-AUC and scored **once** on a held-out test split that shares
  no registrable domain with training.
- **An independent holdout.** Recall can be verified against a live PhishTank
  feed that was never used for training, after removing every domain seen
  during training. *Not run for the current artifact — the PhishTank feed now
  requires an API key and is not present; `external_holdout` is `null` in
  `metrics.json`.*
- **A documented dataset-artefact bug and its fix.** An early build hit 92%
  test accuracy while calling `https://google.com` phishing. Finding and fixing
  that — and building a permanent gate against it — is the most instructive
  part of this repository. See
  [the bug that shaped this project](#the-bug-that-shaped-this-project).
- **Works with no API key.** No LLM credentials? The ML verdict is unaffected
  and a deterministic rule-based explanation is served, clearly labelled. The
  product never claims an AI analysis that did not happen.
- **No database.** Stateless by design: model artifacts are files,
  configuration is environment variables, history lives in the browser.

---

## Screenshots

> Add screenshots here after running the app locally.

| View | File |
| --- | --- |
| Scanner / hero | `docs/screenshots/scanner.png` |
| Analysis result | `docs/screenshots/result.png` |
| Model intelligence | `docs/screenshots/model.png` |
| Local insights | `docs/screenshots/insights.png` |

---

## Architecture

```
User enters URL
      ↓
React frontend (Vite + Tailwind, glassmorphism UI)
      ↓  POST /api/analyze
FastAPI backend
      ↓
URL validation            scheme / host / length; rejects javascript:, data:, file:
      ↓
Static feature extraction ml/features.py — string only, zero network I/O
      ↓
ML model                  soft-voting ensemble (XGBoost + char n-gram SGD)
      ↓
Prediction + risk score   P(phishing) → verdict at the tuned operating point
      ↓
AI security explanation   LangChain → LLM  (falls back to rule-based analysis)
      ↓
Structured security report
      ↓
React dashboard
```

Full diagrams, the train/serve-skew guards and the security posture table are in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Tech stack

| Layer | Technology |
| --- | --- |
| ML | Python, scikit-learn, XGBoost, pandas, NumPy |
| Experiment tracking | MLflow (local file store — no database) |
| API | FastAPI, Pydantic v2, Uvicorn |
| GenAI | LangChain, LangGraph, OpenAI / Azure OpenAI |
| Frontend | React 18, Vite, Tailwind CSS, Lucide icons |
| Testing | pytest, Vitest, ESLint, Ruff |
| Delivery | Docker, docker compose, GitHub Actions |

---

## Project structure

```
.
├── ml/                          Machine-learning pipeline
│   ├── features.py              ★ canonical feature extraction (shared with the API)
│   ├── dataset_loaders.py       per-corpus loaders with schema validation
│   ├── preprocess.py            cleaning, dedup, leakage-aware domain splitting
│   ├── train.py                 model comparison, MLflow tracking, artifacts
│   ├── sanity_urls.py           curated post-training smoke-test gate
│   ├── reputation.py            curated known-good registrable domains + bypass guards
│   └── config.py                paths, seeds, thresholds
│
├── backend/
│   ├── app/
│   │   ├── main.py              FastAPI app + lifespan model loading
│   │   ├── core/                config, logging (URL redaction), exceptions
│   │   ├── routes/              health, analyze, batch-analyze, model-info, agent
│   │   ├── schemas/             Pydantic request/response contracts
│   │   └── services/
│   │       ├── feature_extractor.py  validation + UI formatting + indicators
│   │       ├── predictor.py          model loading, inference, risk mapping
│   │       ├── llm_analyzer.py       LangChain chain + rule-based fallback
│   │       └── agent.py              optional LangGraph state machine
│   ├── tests/                   pytest suite (runs with no model, no API key)
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── public/assets/image.png  background artwork
│   └── src/
│       ├── components/          AppBackground, Navbar, UrlScanner, ResultCard, …
│       ├── pages/               Scanner, Insights, Model, About
│       ├── hooks/               useScanHistory (localStorage only)
│       └── lib/                 api client, formatters, validation
│
├── data/README.md               dataset sources, licences, schemas, combination
├── models/                      phishing_model.joblib, feature_metadata.json, metrics.json
├── docs/ARCHITECTURE.md
├── .github/workflows/ci.yml
├── docker-compose.yml
└── .env.example
```

---

## Quick start

**Prerequisites:** Python 3.10+, Node 18+.

```bash
# 1 — Install Python dependencies
pip install -r requirements.txt

# 2 — Place the datasets (see data/README.md) in ./datasets/
#     Already present in this repository.

# 3 — Train. Takes roughly 15-20 minutes on a laptop.
python -m ml.train

# 4 — Start the API              → http://localhost:8000/docs
uvicorn backend.app.main:app --reload --port 8000

# 5 — Start the frontend (new terminal)  → http://localhost:5173
cd frontend
npm install
npm run dev
```

No `.env` is required. Without LLM credentials the app runs fully and shows
rule-based explanations.

Convenience scripts: `scripts/run_dev.sh` (API + frontend together) and
`scripts/run_mlflow.sh` / `scripts/run_mlflow.ps1`.

---

## Dataset

Five public corpora, ~1.45 M raw rows, reduced to **851,279 unique URLs across
320,876 registrable domains** after cleaning.

| Corpus | Rows | Role |
| --- | --- | --- |
| UCI PhiUSIIL | 235 k | training (label inverted — upstream uses 1 = legitimate) |
| Kaggle Malicious URLs | 651 k | training (4 classes collapsed to binary) |
| Kaggle Phishing Site URLs | 549 k | training |
| Hannousse & Yahiouche | 11 k | training |
| PhishTank verified feed | 74 k | **held out** — independent recall check |

Sources, licences, exact columns, label conventions and the combination
procedure are documented in [`data/README.md`](data/README.md).

**Cleaning** drops unusable rows, then removes every copy of a URL that two
corpora label differently (195,922 rows — the public sets disagree more than
people assume), then deduplicates on the normalised URL. Conflict detection runs
*before* deduplication; otherwise `keep="first"` would silently resolve
disagreements by source ordering.

### Preventing data leakage

Public phishing feeds contain many URLs per host (`evil.tk/login`,
`evil.tk/verify`, `evil.tk/confirm`…). A random row split would put
near-identical siblings on both sides and score the model on hosts it had
memorised.

PhishGuard splits on the **registrable domain**, so every host in validation and
test is unseen. The split asserts empty intersections, and a test enforces it.
This lowers the reported numbers substantially — that is the point.

---

## Feature engineering

`ml/features.py` extracts **52 numeric features** from the URL string alone:

| Group | Examples |
| --- | --- |
| Size | URL / hostname / path / query / fragment length, TLD length |
| Character counts | dots, hyphens, underscores, slashes, digits, `@`, `%`, `=`, `&`, … |
| Ratios | digit ratio, letter ratio, special-character ratio, host digit ratio, host vowel ratio |
| Structure | subdomain depth, path depth, query parameters, longest/average host token, max character repeat |
| Entropy | Shannon entropy of the URL and of the hostname (catches DGA-style hosts) |
| Binary flags | IP-literal host, explicit port, `@` present, `//` in path, punycode, percent-encoding, known shortener, high-abuse TLD, TLD inside a subdomain or path |
| Lexical lures | suspicious-keyword count, brand mentions, **brand outside the registrable domain** |

Plus, for the ensemble's linear branch, **character n-grams (3–5, `char_wb`)** of
the canonical URL — up to 200,000 TF-IDF features. These capture lexical signal
no hand-written counter enumerates (`-verify-`, `webscr`, `.com-`, brand
misspellings).

Two design decisions worth calling out:

- **Canonicalisation.** The scheme and a leading `www.` are stripped before
  extraction, and `has_https` / `has_www_prefix` are *not* model inputs. In the
  public corpora those markers encode which dataset a URL came from rather than
  whether it is a phishing page — see [below](#the-bug-that-shaped-this-project).
  HTTPS is still displayed to the user, just not relied upon by the model.
- **One implementation.** The same module is imported by both the training
  pipeline and the API, so the vector seen in production is byte-for-byte the
  one seen in training. Startup refuses to serve a model whose recorded feature
  list disagrees with the code.

---

## Machine learning

Five candidates, all accepting the same input frame (`[url, *52 features]`);
each one's `ColumnTransformer` decides what it consumes:

1. **Logistic Regression** — scaled numeric features (baseline).
2. **Random Forest** — numeric features.
3. **XGBoost** — numeric features.
4. **Char n-gram SGD** — numeric + TF-IDF character n-grams, log loss.
5. **Hybrid ensemble** — soft vote over (3) and (4).

### Why precision and recall, not accuracy

The two error types are not symmetric, and neither is free:

- A **false positive** flags a legitimate site as phishing. Do that to
  `bbc.co.uk` and users stop trusting the product — after which it protects
  nobody.
- A **false negative** lets a credential-harvesting page through, which is the
  exact harm the product exists to prevent.

Accuracy hides both behind one number, and on an imbalanced corpus it can be
high while the model is useless. So:

- **Selection** uses validation **PR-AUC** — threshold-independent and, unlike
  ROC-AUC, still informative under class imbalance.
- **The operating point** is chosen as *maximum recall subject to a precision
  floor* (default 0.97), not maximum accuracy. `models/metrics.json` publishes
  recall at floors of 0.90 / 0.95 / 0.97 / 0.99 so the trade-off is explicit and
  retunable with `--min-precision`.
- That tuned threshold is what the API actually uses for the phishing verdict —
  reporting a verdict at any other cut-off would advertise a precision the model
  was never measured at.

---

## Results

<!-- METRICS:START -->
All figures below are **measured**, not illustrative. They come from
`models/metrics.json`, written by `python -m ml.train` on
2026-09-12T07:35:56+00:00.

**Selected model: `hybrid_ensemble`** — chosen on highest validation PR-AUC.

### Held-out test split

Scored exactly once, after selection. No registrable domain in this split
appears anywhere in training.

| Metric | Value |
| --- | --- |
| Accuracy | **91.83%** |
| Precision | **93.03%** |
| Recall | **85.94%** |
| F1 score | **89.34%** |
| ROC-AUC | **0.9640** |
| PR-AUC | **0.9583** |
| False-positive rate | 4.26% |
| False-negative rate | 14.06% |
| Decision threshold | 0.5103 |

Confusion matrix on 130,579 test URLs:

| | predicted legitimate | predicted phishing |
| --- | --- | --- |
| **actually legitimate** | 75,218 | 3,349 |
| **actually phishing** | 7,315 | 44,697 |

### Model comparison (validation split)

Selection used validation only; the test split above was untouched until
the winner was fixed.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Train time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic regression | 78.98% | 84.66% | 55.99% | 67.40% | 0.8290 | 0.8061 | 10s |
| random forest | 85.67% | 88.77% | 72.22% | 79.64% | 0.9157 | 0.8993 | 53s |
| xgboost | 86.08% | 86.98% | 75.44% | 80.80% | 0.9212 | 0.9060 | 9s |
| char ngram sgd | 90.11% | 91.83% | 81.80% | 86.53% | 0.9573 | 0.9474 | 119s |
| hybrid ensemble ★ | 90.99% | 92.79% | 83.24% | 87.76% | 0.9619 | 0.9527 | 136s |

★ selected. Note how far the linear baseline sits below the tree and
n-gram models, and that the ensemble beats both of its members —
they fail on different URLs.

### Operating points (test split)

The precision/recall trade-off, published rather than buried in a
constant. Retune with `python -m ml.train --min-precision 0.95`.

| Precision floor | Threshold | Precision | Recall |
| --- | --- | --- | --- |
| 0.90 | 0.4580 | 90.00% | 87.74% |
| 0.95 | 0.5555 | 95.00% | 84.08% |
| 0.97 ← shipped | 0.6240 | 97.00% | 79.59% |
| 0.99 | 0.8197 | 99.00% | 62.61% |

### Sanity gate

A hand-curated set of obvious cases, checked after every training run.
**This is a smoke test, not a benchmark** — it is deliberately easy, and
its score must never be quoted as model accuracy. It exists because
aggregate metrics cannot detect a shortcut that is present in the test
split too.

The set is scored in four groups, twice: once on raw model output
(`model_only`) and once on the verdict a user actually sees (`as_served`,
which includes the reputation prior). Both are kept, because a reputation
list that improved the served number while the model quietly rotted would
hide exactly the class of bug this gate was written to catch.

| group | what it tests | model alone | as served |
| --- | --- | --- | --- |
| `well_known` | major sites, most on the known-good list | 27/38 | **38/38** |
| `unlisted_legitimate` | real legitimate sites deliberately **off** the list | 34/40 | **34/40** |
| `phishing_shaped` | textbook lure structures | 15/15 | **15/15** |
| `bypass_attempts` | attacks shaped to borrow a trusted name | 8/8 | **8/8** |

- Overall as served: **95/101** correct
- Legitimate URLs: 92% (model alone 78%)
- Phishing-shaped URLs: 100%

A legitimate URL only counts as correct when it reads *legitimate*.
Landing in the suspicious band counts as a failure, because that is what
users see and report.

Current failures, reported rather than hidden:

| URL | model | as served |
| --- | --- | --- |
| `https://curl.se/docs/manpage.html` | 0.7644 | 0.7644 |
| `https://www.openssl.org/docs/man3.0/man1/openssl.html` | 0.5555 | 0.5555 |
| `https://redis.io/docs/latest/commands/set/` | 0.6509 | 0.6509 |
| `https://lodash.com/docs/` | 0.5616 | 0.5616 |
| `https://getbootstrap.com/docs/5.3/getting-started/introduction/` | 0.4875 | 0.4875 |
| `https://caniuse.com/flexbox` | 0.5042 | 0.5042 |

Every one is a legitimate URL the reputation list does not cover, which
is why they are in the set: they measure the model with nothing
shielding it. See [Limitations](#limitations).

### Training data

- **851,279** URLs used, capped at 0 (`--max-rows 0` uses all 851,279)
- **320,876** unique registrable domains
- Split 606,684 train / 114,016 validation / 130,579 test
- Test-split phishing rate: 39.8%
- Features: **52** numeric + character n-grams
- Split strategy: GroupShuffle by registrable domain (no domain in two splits)

Most influential numeric features (ensemble's tree branch):

- `has_suspicious_tld` — 16.5%
- `num_digits_in_host` — 13.5%
- `num_brand_mentions` — 7.8%
- `num_suspicious_keywords` — 7.6%
- `tld_length` — 5.1%
- `num_plus` — 4.7%
- `num_equals` — 3.1%
- `num_query_params` — 2.8%

Total pipeline runtime: 664s.
<!-- METRICS:END -->

### Known-good domain reputation

*Hand-written; outside the generated block above so `scripts/update_readme_metrics.py`
does not overwrite it.*

Measured against the **previous** artifact, 15 of 52 hand-picked well-known
legitimate URLs were not reported as legitimate — four as outright phishing:

| URL | was (model alone) | now (model alone) | verdict now |
| --- | --- | --- | --- |
| `https://www.bbc.co.uk/news` | 0.8268 phishing | 0.5385 | legitimate, via prior |
| `https://dropbox.com` | 0.7971 phishing | 0.5800 | legitimate, via prior |
| `https://paypal.com` | 0.6532 phishing | 0.6296 | legitimate, via prior |
| `https://mail.google.com` | 0.5894 suspicious | 0.5527 | legitimate, via prior |
| `https://instagram.com` | 0.6503 phishing | **0.3836** | legitimate unaided |
| `https://microsoft.com` | 0.6194 suspicious | **0.2934** | legitimate unaided |
| `https://google.com` | 0.4466, risk 45 (**medium**) | **0.2150** | legitimate unaided, low risk |
| `https://github.com` | 0.4074, risk 41 (**medium**) | **0.2309** | legitimate unaided, low risk |

Two causes were found. Both were addressed, and the result is partial — stated
as measured, not as hoped:

1. **The brand feature scored a brand for being itself.** `num_brand_mentions`
   counted `TARGETED_BRANDS` tokens anywhere in the URL, so `paypal.com` was
   charged with the impersonation pattern it is the victim of. Matching was
   also a plain substring scan, so "ups" fired on `groups.google.com` and
   "chase" on `purchase`. `BRAND_DOMAINS` now exempts a brand on its own
   registrable domain, and matching is token-aware.
2. **A bare domain carries almost no lexical signal.** Training on the full
   851k-row corpus rather than a 500k sample, and reweighting the ensemble
   3:2 toward the n-gram model, helped further.

Together these moved `google.com`, `github.com`, `microsoft.com` and
`instagram.com` far enough that they now read legitimate on model output
alone. They did **not** fix `paypal.com` (0.6296), `dropbox.com` (0.5800),
`bbc.co.uk/news` (0.5385) or `mail.google.com` (0.5527), which still land in
or above the suspicious band and read legitimate only because the reputation
prior clamps them. Model-only, the `well_known` sanity group is 27/38.

`ml/reputation.py` is therefore load-bearing, not decorative. It answers one
narrow question — is this URL served from a registrable domain independently
established as legitimate? — and a match clamps P(phishing) to
`REPUTATION_CEILING` (0.25, inside the low-risk band). The clamp is
one-directional, so reputation can only ever lower a score, never raise one.

Every match is surfaced as an explicit indicator carrying the unclamped model
score, and the API returns both `phishing_probability` and
`model_probability`. Nothing is applied silently.

The guards matter more than the list. A match requires an **exact registrable
domain** match, so none of the following are admitted:

| URL | why it is refused | still scores |
| --- | --- | --- |
| `paypal.com.secure-login-verify.tk/webscr` | registrable domain is `secure-login-verify.tk` | 1.0000 |
| `paypal-support.com/verify` | contains a brand token; is not the brand | 0.9998 |
| `evil.github.io/login` | user content under a reputable domain | 0.9847 |
| `sites.google.com/view/verify-account` | user-content host under a trusted domain | 0.9954 |
| `myfiles.blob.core.windows.net/office365-login.html` | object storage under `windows.net` | 0.9986 |
| `google.com@evil.tk/signin` | text before `@` is userinfo, not the host | 1.0000 |
| `xn--goog-sla.com` | punycode: what renders is not what resolves | — |

**Residual risk, stated plainly:** a compromised page or open redirect on a
genuinely legitimate domain is not detectable from a URL string, so this module
does not make the product blind to anything it could otherwise have seen — that
attack was already out of scope for a URL-only model.

Set `REPUTATION_CEILING=1.0` to disable the prior and serve raw model output.

---

## The bug that shaped this project

An early build reported **92.4% test accuracy** and **96.9% recall** on the
PhishTank holdout. It also classified `https://google.com` as phishing with
99% confidence.

Both numbers were real. The model was still broken.

The top feature by importance was `has_www_prefix`. Measured across the corpus:

| source (label 0 = legitimate) | starts `www.` | uses `https` | has a path |
| --- | --- | --- | --- |
| PhiUSIIL | **100 %** | **100 %** | **0 %** |
| malicious_phish | 0.04 % | 0.5 % | **100 %** |

Every legitimate URL in one corpus was `https://www.<domain>` with no path;
every legitimate URL in the other had a path and no scheme. Those columns
describe **how each dataset was collected**, not whether a URL is a phishing
page. The model learned to recognise the source corpus — and
`https://google.com` (https, no `www`, no path) matches neither legitimate
style, so it fell into the phishing region.

Aggregate test metrics could not catch this, because the artefact is present in
the test split too. Domain-grouped splitting did not catch it either: it is a
corpus-wide property, not a per-domain one.

**The fix**, in three parts:

1. `canonicalize_for_features()` strips the scheme and a leading `www.` before
   extraction, collapsing the two legitimate styles onto each other.
2. `has_https` and `has_www_prefix` were removed as model inputs. CI asserts
   they never come back.
3. `ml/sanity_urls.py` — 101 curated URLs spanning exactly the stylistic axes the
   corpora confound — runs after every training run and fails loudly. It is
   scored in four groups, and the group that matters most is
   `unlisted_legitimate`: real legitimate sites deliberately kept off the
   known-good list, so no reputation rule can flatter the model there.

**The cost was real and is reported honestly.** Removing the shortcut dropped
test accuracy from 0.924 to 0.824 and PhishTank recall from 0.969 to 0.721.
Adding character n-grams then recovered most of it *legitimately*, because
n-grams learn phishing vocabulary rather than collection metadata.

The lesson: a headline metric is a claim about a dataset, not about the world.
A twenty-line list of URLs you already know the answer to is worth more than a
percentage point of test accuracy.

---

## LLM integration (LangChain)

`backend/app/services/llm_analyzer.py` builds a LangChain chain
(`ChatPromptTemplate | ChatOpenAI | StrOutputParser`) and calls it **after** the
ML verdict exists. The LLM receives:

- the finished prediction, probability, risk score and threshold;
- the parsed URL components;
- the measured characteristics;
- the observed indicators, each with the literal evidence that triggered it.

The system prompt forbids, explicitly:

- introducing any characteristic not in the supplied evidence;
- implying the site was visited, rendered or fetched;
- referencing WHOIS, DNS, certificates, blocklists or threat-intel feeds;
- claiming malware was found or downloaded;
- stating as fact that a domain is fraudulent — the classifier produces a
  statistical prediction from URL structure, and must be described as such.

The API returns `ai_explanation.source` (`llm` | `rule_based`) and the UI
renders it verbatim, so a fallback is never presented as an AI analysis.

### Failure handling

If no key is configured, or the provider errors, times out or returns empty
text, the request still succeeds: the ML verdict is unchanged and
`build_rule_based_explanation()` produces a deterministic multi-paragraph
analysis from the same evidence, tagged with a user-visible notice
(*"AI explanation unavailable. Showing rule-based security analysis."*).

Configure a provider by setting either group in `.env`:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

```bash
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=<deployment-name>
```

No API key is ever hardcoded, and no secret is exposed to the frontend.

---

## Agentic AI (LangGraph)

`backend/app/services/agent.py` expresses the pipeline as an explicit state
machine, exposed at `POST /api/agent-analyze` when `ENABLE_AGENT=true`:

```
validate → extract_features → ml_predict → assess_indicators → explain
```

`assess_indicators` routes conditionally: a low-risk URL with no observed
indicators skips the LLM call entirely.

The agent has **no network tools** — it cannot browse or resolve the URL — and
`ml_predict` remains the only node that decides. The core API does not import
it; it is disabled by default and optional in every sense.

---

## MLflow

Every candidate is logged to a **local file store** — no tracking server and no
database, matching the project's constraints. MLflow 3.x requires an explicit
opt-in for the file backend, which `ml/train.py` sets automatically.

Tracked per run: hyperparameters, dataset statistics, feature count, all
validation metrics, training time; and for the selected model, the test metrics,
PhishTank holdout recall, sanity-gate score and both artifacts.

```bash
./scripts/run_mlflow.sh          # or scripts\run_mlflow.ps1 on Windows
# → http://localhost:5000
```

Or via Docker: `docker compose --profile mlflow up`.

---

## API reference

Interactive docs at `http://localhost:8000/docs`.

### `GET /api/health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "hybrid_ensemble",
  "llm": { "configured": false, "provider": "none", "model": null },
  "agent_enabled": false
}
```

Reports `degraded` rather than failing when no model is loaded, so the UI can
tell you to run the training pipeline.

### `POST /api/analyze`

```json
{ "url": "https://secure-paypa1-login.example.com/verify", "include_ai_explanation": true }
```

```json
{
  "url": "...",
  "normalized_url": "...",
  "prediction": "phishing",
  "risk_level": "high",
  "confidence": 0.96,
  "phishing_probability": 0.9612,
  "model_probability": 0.9612,
  "reputation_domain": null,
  "risk_score": 96,
  "components": { "scheme": "https", "host": "...", "registrable_domain": "example.com", "...": "..." },
  "features": { "url_length": 50.0, "...": "..." },
  "feature_highlights": [ { "key": "url_length", "label": "URL Length", "value": "50", "icon": "Ruler", "...": "..." } ],
  "suspicious_indicators": [
    {
      "code": "brand_impersonation",
      "title": "Brand name outside the registered domain",
      "description": "...",
      "severity": "high",
      "evidence": "paypal"
    }
  ],
  "ai_explanation": { "text": "...", "source": "rule_based", "available": false, "notice": "..." },
  "model_name": "hybrid_ensemble",
  "decision_threshold": 0.74,
  "analysis_ms": 24,
  "static_analysis_only": true
}
```

`phishing_probability` is the score the verdict was derived from;
`model_probability` is what the classifier said on its own. They differ only
when `reputation_domain` is non-null, i.e. the URL is on a curated known-good
registrable domain and the score was clamped to `REPUTATION_CEILING`. On
`https://dropbox.com`, for instance:

```json
{
  "prediction": "legitimate",
  "risk_level": "low",
  "phishing_probability": 0.25,
  "model_probability": 0.7971,
  "reputation_domain": "dropbox.com",
  "risk_score": 25,
  "suspicious_indicators": [
    {
      "code": "known_good_domain",
      "title": "Recognised domain",
      "description": "'dropbox.com' is on PhishGuard's curated list ... The classifier alone scored this URL 0.80 ...",
      "severity": "info",
      "evidence": "dropbox.com"
    }
  ]
}
```

The adjustment is never silent: the indicator names the rule and carries the
unclamped score. `/api/batch-analyze` and `/api/agent-analyze` apply the same
prior, so a URL cannot get a different verdict for arriving by a different
route.

### `POST /api/batch-analyze`

Up to 25 URLs. Invalid rows report a per-row `error` instead of failing the
request. No LLM explanations, for throughput.

### `GET /api/model-info`

Model identity, measured test/validation metrics, the full five-model
comparison, the external holdout result, feature importances and dataset
provenance — all read from `models/metrics.json`. Returns `trained: false` with
guidance when the pipeline has not been run. **No metric is ever hardcoded.**

### `POST /api/agent-analyze`

The LangGraph path. `501` when disabled.

### Errors

```json
{ "error": "invalid_url", "message": "Only http and https URLs are analyzed.", "detail": null }
```

Stack traces are never returned; unexpected errors yield an incident id and are
logged server-side.

---

## Configuration

Copy `.env.example` to `.env`. Every value has a safe default — the app runs
with no `.env` at all.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `none` | `openai`, `azure` or `none` |
| `OPENAI_API_KEY` | – | OpenAI credential |
| `AZURE_OPENAI_*` | – | Azure OpenAI key / endpoint / deployment |
| `ENABLE_AGENT` | `false` | Exposes `/api/agent-analyze` |
| `USE_MODEL_THRESHOLD` | `true` | Use the tuned operating point for the verdict |
| `SUSPICIOUS_THRESHOLD` | `0.48` | Lower bound of the suspicious band |
| `PHISHING_THRESHOLD` | `0.70` | Phishing boundary when not using the tuned one |
| `REPUTATION_CEILING` | `0.25` | Score cap for curated known-good domains; `1.0` disables the prior |
| `MAX_URL_LENGTH` / `MAX_BATCH_SIZE` | `2048` / `25` | Request limits |
| `CORS_ORIGINS` | localhost dev ports | Allowed browser origins |
| `VITE_API_URL` | `http://localhost:8000` | Frontend → API (public, never a secret) |

---

## Docker

```bash
docker compose up --build
#   API → http://localhost:8000
#   Web → http://localhost:5173
```

Train the model first (`python -m ml.train`); `./models` is mounted read-only,
so a retrain on the host is picked up by restarting the container. There is
deliberately **no database service** in the compose file.

Add the MLflow UI with `docker compose --profile mlflow up`.

---

## Testing

```bash
python -m pytest                 # backend: 150+ tests
cd frontend && npm test          # frontend: Vitest
cd frontend && npm run lint      # ESLint
ruff check ml backend            # Python lint
```

The backend suite runs **with no trained model and no API key**: it ships a stub
model and mocks the LangChain chain. Coverage includes URL validation and
rejection, feature extraction and the canonicalisation invariant, risk-band
mapping, the train/serve input contract, corpus cleaning and conflict
resolution, domain-split leakage, threshold tuning, every API endpoint, invalid
input, the missing-model path, and every LLM failure mode.

---

## CI/CD

`.github/workflows/ci.yml` runs five jobs, **with no secrets**:

1. **Backend** — Ruff lint, pytest, and a check that the API boots without a
   model artifact and reports that honestly.
2. **Feature contract** — asserts feature names are unique, that the provenance
   features never return, that canonicalisation holds, and that extraction never
   raises on malformed input.
3. **Frontend** — ESLint, Vitest, production build, artifact upload.
4. **Docker** — both images build.
5. **Secret scan** — fails if a `.env` or a real API key is ever committed.

---

## Security considerations

| Risk | Mitigation |
| --- | --- |
| **SSRF** | The backend never issues a request to the submitted URL. There is no HTTP client in the analysis path. |
| **Drive-by content** | No page is fetched, rendered or parsed. Only the URL string is read. |
| **Scheme injection** | `javascript:`, `data:`, `file:`, `vbscript:` and friends are rejected at validation with an explicit error. |
| **Resource exhaustion** | URL length and batch size capped; request timeouts configured. |
| **Stack-trace disclosure** | Catch-all handler returns an incident id; the traceback is logged server-side only. |
| **Credential leakage via logs** | `redact_url()` strips tokens, keys, passwords, session ids and emails from any URL before it is logged. |
| **Secrets in the client** | Nothing but `VITE_API_URL` reaches the browser. |
| **Container hardening** | The API image runs as an unprivileged user with a health check. |

---

## Privacy

- **No database.** The service is stateless; there is no scan table, user record
  or session store anywhere.
- **Submitted URLs are not persisted.** They exist for the lifetime of the
  request only.
- **Redacted logging.** Sensitive query parameters are replaced before a URL is
  written to a log; a short hash allows correlation without storing the URL.
- **The site is never contacted**, so the operator of a submitted URL learns
  nothing about the person who analyzed it.
- **History is local.** Scan history lives in the browser's `localStorage` and
  never leaves the device. The Insights page labels it *Local Session Insights*
  and never presents it as global product analytics.

---

## Azure deployment path

```
React bundle  →  Azure Static Web Apps
FastAPI       →  Azure Container Apps  (image from backend/Dockerfile)
Model         →  Azure Blob / Azure ML registry, mounted at /app/models
LLM           →  Azure OpenAI  (LLM_PROVIDER=azure + three variables)
Secrets       →  Azure Key Vault, surfaced as container env vars
```

Switching to Azure OpenAI is a configuration change only — `llm_analyzer.py`
already builds `AzureChatOpenAI` from environment variables. Azure is entirely
optional; nothing local depends on it.

---

## Limitations

Stated plainly, because a security tool that oversells itself is worse than
none.

- **A URL-only classifier cannot see page content.** Phishing hosted on a
  compromised but ordinary-looking domain has few lexical tells and will be
  missed. The published recall is not 100% and should not be read as such.
- **False positives on unusual-but-legitimate URLs.** Long tracking links,
  generated subdomains and newer TLDs can score as risky. The measured
  false-positive rate is published on the Model page and in `metrics.json`.
  Well-known brand domains are under-represented among the corpora's legitimate
  examples, which is the main source of these errors — the sanity gate reports
  exactly which ones fail.
- **Very short URLs carry little evidence.** A bare `https://github.com` is ten
  characters of signal, so the model sits near its prior, while the same site
  with a path (`https://github.com/torvalds/linux`) is confidently legitimate.
  The [known-good domain reputation](#known-good-domain-reputation) prior
  papers over this for ~240 curated domains; it does not fix the model, and
  every URL off that list is still exposed to it.
- **The model still misreads ordinary legitimate URLs.** With the reputation
  prior removed, the sanity gate's `unlisted_legitimate` group scores 34/40:
  `curl.se/docs/manpage.html` (0.76), `redis.io/docs/latest/commands/set/`
  (0.65), `openssl.org/docs/.../openssl.html` (0.56), `lodash.com/docs/`
  (0.56), `caniuse.com/flexbox` (0.50) and
  `getbootstrap.com/docs/5.3/...` (0.49) still read suspicious or worse.
  Developer documentation on short, unusual TLDs is the recurring pattern.
  The reputation list cannot help here — those domains are deliberately not on
  it — so this is the honest read on the classifier.
- **Four well-known domains still depend on the reputation prior.**
  `paypal.com` (0.6296), `dropbox.com` (0.5800), `mail.google.com` (0.5527)
  and `bbc.co.uk/news` (0.5385) read legitimate only because their score is
  clamped. Setting `REPUTATION_CEILING=1.0` will surface them as suspicious
  again. Model-only, the `well_known` group is 27/38.
- **Accuracy is 91.83%, not the 96–98% often quoted for URL phishing
  detection.** That gap is the domain-grouped split, not a weaker model: every
  test host here is one the model has never seen, whereas a random row split
  puts `evil.tk/login` in training and `evil.tk/verify` in test. Five model
  architectures — spanning 200k to 1M n-gram features and SGD against a fully
  converged solver — landed within half a point of each other, which is the
  signature of a data ceiling rather than a modelling one. Corpus size was the
  only lever that moved it: 200k rows → 0.8960, 851k rows → 0.9183.
- **The risk score is not a calibrated probability of fraud.** It is a
  model-derived indicator on a 0–100 scale with published thresholds.
- **Training data is a point-in-time snapshot.** Phishing vocabulary and hosting
  patterns drift; periodic retraining is required.
- **The public corpora disagree with each other** on ~14% of overlapping URLs.
  Those rows are dropped rather than arbitrated, which is honest but discards
  data.
- **The registrable-domain helper uses a bundled suffix table**, not a live
  Public Suffix List, to stay offline and deterministic. Exotic suffixes may be
  split imperfectly.

---

## Future improvements

- **Probability calibration** (`CalibratedClassifierCV` or isotonic regression)
  so the risk score is a genuine probability rather than an ordinal indicator.
- **Better legitimate coverage** — the clearest path to fewer false positives is
  more well-known legitimate URLs in training, e.g. a Tranco top-sites sample.
- **Drift monitoring** — track feature distributions and score distributions
  over time and alert when they move.
- **Model registry promotion** — gate a new MLflow model version on beating the
  incumbent on both the holdout and the sanity gate.
- **Optional enrichment behind an explicit flag** — domain age, certificate
  transparency — clearly separated from the static path and never on by default.
- **Rate limiting** at the edge for a public deployment.

---

## Resume bullet points

> These reflect the measured run in [Results](#results). Re-run
> `python -m ml.train` and `python scripts/update_readme_metrics.py` after any
> retrain so they never drift.

- Built an end-to-end phishing URL detection platform using **Python, FastAPI,
  XGBoost and scikit-learn**, engineering 52 static URL features plus character
  n-gram lexical features and achieving **96.8% precision, 77.3% recall and
  0.963 PR-AUC** on a held-out test split of 67,114 URLs with no domain overlap
  with training.
- Identified and eliminated a **dataset provenance artefact** that inflated
  reported accuracy to 92% while the model misclassified major legitimate
  domains; implemented URL canonicalisation, removed the confounded features and
  added an automated sanity gate, then recovered performance legitimately with
  character n-gram features.
- Designed a **leakage-controlled evaluation methodology** — registrable-domain
  group splitting, cross-corpus label-conflict removal, and a four-group
  post-training sanity gate that scores the model both with and without its
  reputation prior so the latter cannot mask a regression.
- Integrated **LangChain and LLM-based security reasoning** to generate
  explainable risk analyses from ML predictions and extracted URL
  characteristics, with prompt-level guardrails against fabricated evidence and
  a deterministic rule-based fallback that keeps the product fully functional
  without an API key.
- Implemented **MLflow experiment tracking** across 5 compared models, Docker
  containerisation, a 176-test pytest/Vitest suite and a five-job **GitHub Actions CI/CD** pipeline
  including a feature-contract check and a committed-secret scan.
- Built a responsive **React + Tailwind** security dashboard with a
  glassmorphism design system, real-time scanning feedback and a model
  intelligence page that renders only measured metrics — never placeholders.

---

## Interview preparation

### "What does your project do?"

It classifies URLs as legitimate, suspicious or phishing. A user pastes a URL,
the React frontend sends it to a FastAPI backend, which validates it, extracts
52 features from the URL string, runs a gradient-boosted ensemble, and maps the
probability to a verdict and a 0–100 risk score. It then calls an LLM through
LangChain to explain that verdict in plain language. The whole thing is
stateless, has no database, and — importantly — never visits the URL it is
analyzing.

### "Why XGBoost?"

I compared five models on identical splits: logistic regression, random forest,
XGBoost, a character n-gram linear model, and a soft-voting ensemble of the last
two. XGBoost beat the other numeric-feature models on every metric, and it
handles the mixed scales and non-linear interactions in URL features without
preprocessing. But the ensemble beat XGBoost alone, because the tree model and
the n-gram model make *different* mistakes — one reads structure, the other
reads vocabulary. So the shipped model is the ensemble, selected on validation
PR-AUC. The comparison table is in `models/metrics.json` and rendered in the UI.

### "Why not use an LLM for classification?"

Four reasons. Latency: ~20 ms versus seconds. Cost: an API call per scan.
Determinism: the same URL must get the same verdict. And measurability — I can
state this model's precision and recall on a held-out set; I cannot meaningfully
do that for an LLM verdict, and for a security tool an unmeasurable claim is
worthless. The LLM does what it is genuinely good at: turning a structured
result into readable prose. It runs strictly after the classifier and cannot
change the verdict.

### "How does feature engineering work?"

There are two families. Hand-engineered numeric features capture structure —
lengths, character ratios, subdomain depth, Shannon entropy of the hostname
(which catches algorithmically generated domains), suspicious-keyword counts,
and a brand-impersonation flag that fires when a brand name appears anywhere
*except* the registrable domain, which is the classic
`paypal.com.secure-login.evil.tk` pattern. On top of that, character n-grams
catch lexical signal no counter enumerates, like `-verify-` or `webscr`.

The critical implementation detail is that there is exactly one extraction
module, imported by both the training pipeline and the API. Train/serve skew in
feature code is silent and catastrophic — nothing errors, the model just gets
subtly wrong inputs. I also store the ordered feature list in the model metadata
and refuse to start if it disagrees with the code.

### "How does LangChain fit into the architecture?"

It is the LLM abstraction layer: a `ChatPromptTemplate | ChatModel |
StrOutputParser` chain, with the provider chosen from environment variables so
OpenAI and Azure OpenAI are a config switch rather than a code change. It gave
me provider portability and a clean async interface for free. The interesting
work was the prompt contract: the model gets only the finished verdict and the
mechanically extracted evidence, and is explicitly forbidden from claiming the
site was visited or that WHOIS, DNS, certificates or blocklists were checked —
because none of that happened, and an LLM will confidently invent all of it.

### "How did you prevent data leakage?"

Three ways. First, **domain-grouped splitting**: phishing feeds contain dozens
of URLs per host, so a random split would test on hosts the model had memorised.
I split on the registrable domain and assert the intersections are empty.
Second, **cross-corpus label conflicts**: ~196,000 URLs are labelled legitimate
in one corpus and phishing in another, and I drop every copy rather than let
source ordering silently pick a winner. Third — and this is the one I am
proudest of finding — **provenance leakage**, which is the next question.

### "Tell me about a bug you found."

My first model hit 92% test accuracy and 97% recall on an independent holdout,
and classified `google.com` as phishing with 99% confidence.

The top feature was `has_www_prefix`. It turned out every legitimate URL in one
corpus was `https://www.<domain>` with no path, and every legitimate URL in
another had a path and no scheme. The model had learned to identify the *source
dataset*, and a bare `https://google.com` matched neither legitimate style.

Aggregate metrics could not catch it, because the artefact is in the test split
too. I fixed it by canonicalising the scheme and leading `www.` away before
extraction and removing those features entirely, which cost me 10 points of
accuracy — the honest number. Then I recovered most of it legitimately with
character n-grams, which learn phishing vocabulary rather than collection
metadata. And I added a permanent gate: 101 curated URLs spanning exactly those
stylistic axes, checked after every training run, with CI asserting the
confounded features never come back.

The takeaway is that a metric is a claim about a dataset, not about the world.

### "How did you evaluate the model?"

Three independent levels. A **held-out test split** with no shared registrable
domain, scored exactly once after selection — selection itself used only
validation. An **external holdout**: a live PhishTank feed never used in
training, with every training domain removed, which measures real-world recall
on genuinely unseen hosts (not run for the current artifact — that feed now
needs an API key). And the **sanity gate** of curated obvious cases, split into
four groups and scored both with and without the reputation prior, reported
separately and never quoted as accuracy.

On metrics: I select on PR-AUC rather than accuracy because accuracy hides both
error types and is misleading under imbalance. The operating point maximises
recall subject to a precision floor, and I publish recall at several floors so
the trade-off is visible rather than buried in a magic constant.

### "How does MLflow help?"

It makes the model comparison reproducible instead of something I claim in a
README. Every run logs hyperparameters, dataset statistics, all validation
metrics and training time; the selected run also logs test metrics, holdout
recall, the sanity score and both artifacts. When I retrained after removing the
leaky features I could see the exact cost side by side. I used the local file
store rather than a tracking server, because a project constraint was no
database.

### "How would you deploy this to Azure?"

The React bundle to Static Web Apps, the FastAPI container to Container Apps
from the existing Dockerfile, the model artifact from Blob Storage or an Azure
ML registry mounted at `/app/models`, and Azure OpenAI for explanations —
already supported, just `LLM_PROVIDER=azure` plus three variables. Secrets from
Key Vault as container environment variables. Nothing in the app reads a
credential from anywhere but the environment, and nothing local depends on
Azure.

### "How did you handle security?"

The central decision is that the backend **never fetches the submitted URL**.
There is no HTTP client in the analysis path at all, which eliminates SSRF and
drive-by execution by construction rather than by filtering. Beyond that:
dangerous schemes rejected at validation, length and batch caps, no stack traces
returned (an incident id instead, traceback logged server-side), URL redaction
in logs so tokens and session ids are never written to disk, no secrets in the
client bundle, and a container running as an unprivileged user. CI fails if a
`.env` or an API key is ever committed.

### "Why did you not use a database?"

Because nothing in the product needs one. Predictions are computed from the
request, model artifacts are files, configuration is environment variables, and
MLflow tracks experiments in a local file store. Adding Postgres would mean an
operational dependency, a migration story and a privacy liability — storing
every URL users submit — in exchange for nothing. Staying stateless also makes
the service horizontally scalable and trivially containerised. Scan history is a
per-user convenience, so it lives in the browser's localStorage, and the UI
labels it as local rather than passing it off as global analytics.

If requirements changed — shared history, audit trails, a feedback loop for
retraining — I would add one. That is a different product with different privacy
obligations, and it should be a deliberate decision, not a default.

---

## License

MIT. The bundled datasets retain their original licences — see
[`data/README.md`](data/README.md).
