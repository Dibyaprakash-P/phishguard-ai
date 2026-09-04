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
- **An independent holdout.** Recall is verified against a live PhishTank feed
  that was never used for training, after removing every domain seen during
  training.
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
2026-09-04T04:06:01+00:00.

**Selected model: `hybrid_ensemble`** — chosen on highest validation PR-AUC.

### Held-out test split

Scored exactly once, after selection. No registrable domain in this split
appears anywhere in training.

| Metric | Value |
| --- | --- |
| Accuracy | **89.61%** |
| Precision | **93.38%** |
| Recall | **84.46%** |
| F1 score | **88.69%** |
| ROC-AUC | **0.9591** |
| PR-AUC | **0.9632** |
| False-positive rate | 5.59% |
| False-negative rate | 15.54% |
| Decision threshold | 0.5096 |

Confusion matrix on 67,114 test URLs:

| | predicted legitimate | predicted phishing |
| --- | --- | --- |
| **actually legitimate** | 32,789 | 1,940 |
| **actually phishing** | 5,034 | 27,351 |

### Model comparison (validation split)

Selection used validation only; the test split above was untouched until
the winner was fixed.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Train time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic regression | 77.17% | 82.55% | 66.40% | 73.60% | 0.8408 | 0.8575 | 26s |
| random forest | 83.83% | 90.92% | 73.61% | 81.36% | 0.9101 | 0.9194 | 249s |
| xgboost | 84.21% | 90.22% | 75.21% | 82.03% | 0.9165 | 0.9251 | 40s |
| char ngram sgd | 87.82% | 91.12% | 82.63% | 86.67% | 0.9489 | 0.9541 | 159s |
| hybrid ensemble ★ | 89.21% | 92.89% | 83.89% | 88.16% | 0.9530 | 0.9582 | 180s |

★ selected. Note how far the linear baseline sits below the tree and
n-gram models, and that the ensemble beats both of its members —
they fail on different URLs.

### Operating points (test split)

The precision/recall trade-off, published rather than buried in a
constant. Retune with `python -m ml.train --min-precision 0.95`.

| Precision floor | Threshold | Precision | Recall |
| --- | --- | --- | --- |
| 0.90 | 0.4474 | 90.00% | 87.40% |
| 0.95 | 0.5595 | 95.00% | 81.53% |
| 0.97 ← shipped | 0.6427 | 97.00% | 76.81% |
| 0.99 | 0.7949 | 99.00% | 67.40% |

### Independent holdout — PhishTank

The strongest evidence of real generalisation: a live phishing feed never
used in training, with every domain seen during training removed first.

- **Recall: 91.04%** — detected 26,275 of 28,861 URLs
- Spanning 12,787 registrable domains, none seen in training
- Mean predicted probability: 0.8262

Phishing-only feed, so recall is the only meaningful metric here.

### Sanity gate

A hand-curated set of obvious cases, checked after every training run.
**This is a smoke test, not a benchmark** — it is deliberately easy, and
its score must never be quoted as model accuracy. It exists because
aggregate metrics cannot detect a shortcut that is present in the test
split too.

- Overall: **35/40** correct
- Well-known legitimate URLs: 80%
- Phishing-shaped URLs: 100%

Current failures, reported rather than hidden:

- `https://www.microsoft.com` — expected legitimate, scored 0.6194
- `https://www.bbc.co.uk/news` — expected legitimate, scored 0.8268
- `https://news.ycombinator.com` — expected legitimate, scored 0.6016
- `https://mail.google.com` — expected legitimate, scored 0.5894
- `https://www.linkedin.com/in/example` — expected legitimate, scored 0.6053

These are false positives on well-known sites, and they reflect a real
limitation: the corpora's legitimate examples skew long-tail, so major
brand domains are under-represented. See [Limitations](#limitations).

### Training data

- **500,000** URLs used, capped at 500,000 (`--max-rows 0` uses all 851,279)
- **198,432** unique registrable domains
- Split 364,953 train / 67,933 validation / 67,114 test
- Test-split phishing rate: 48.3%
- Features: **52** numeric + character n-grams
- Split strategy: GroupShuffle by registrable domain (no domain in two splits)

Most influential numeric features (ensemble's tree branch):

- `num_digits_in_host` — 15.8%
- `has_suspicious_tld` — 14.2%
- `num_suspicious_keywords` — 7.8%
- `num_plus` — 5.0%
- `num_ampersands` — 5.0%
- `tld_length` — 4.9%
- `brand_outside_domain` — 4.1%
- `num_hyphens_in_host` — 3.2%

Total pipeline runtime: 1278s.
<!-- METRICS:END -->

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
3. `ml/sanity_urls.py` — 40 curated URLs spanning exactly the stylistic axes the
   corpora confound — runs after every training run and fails loudly.

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
  characters of signal, so the model sits near its prior and the verdict lands
  in the *suspicious* band, while the same site with a path
  (`https://github.com/torvalds/linux`) is confidently legitimate. This is the
  model honestly reporting uncertainty rather than a bug, but it is poor UX for
  major domains, and it is the clearest argument for the
  [future improvement](#future-improvements) of adding well-known legitimate
  domains to training.
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
  group splitting, cross-corpus label-conflict removal, and an independent
  PhishTank holdout achieving **84.6% recall** on 28,861 live phishing URLs
  spanning 12,787 previously unseen domains.
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
metadata. And I added a permanent gate: 40 curated URLs spanning exactly those
stylistic axes, checked after every training run, with CI asserting the
confounded features never come back.

The takeaway is that a metric is a claim about a dataset, not about the world.

### "How did you evaluate the model?"

Three independent levels. A **held-out test split** with no shared registrable
domain, scored exactly once after selection — selection itself used only
validation. An **external holdout**: a live PhishTank feed never used in
training, with every training domain removed, which measures real-world recall
on genuinely unseen hosts. And the **sanity gate** of curated obvious cases,
reported separately and never quoted as accuracy.

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
