# Architecture

## Request flow

```
                          ┌──────────────────────────────┐
   User enters a URL ───► │  React + Vite frontend       │
                          │  (glass UI, Tailwind)        │
                          └───────────────┬──────────────┘
                                          │  POST /api/analyze  { "url": "..." }
                                          ▼
                          ┌──────────────────────────────┐
                          │  FastAPI backend             │
                          └───────────────┬──────────────┘
                                          ▼
                    ┌─────────────────────────────────────────┐
                    │ 1. URL validation                       │
                    │    scheme / host / length / charset     │
                    │    rejects javascript:, data:, file:    │
                    └─────────────────────┬───────────────────┘
                                          ▼
                    ┌─────────────────────────────────────────┐
                    │ 2. Static feature extraction            │
                    │    ml/features.py — string only,        │
                    │    NO network I/O of any kind           │
                    └─────────────────────┬───────────────────┘
                                          ▼
                    ┌─────────────────────────────────────────┐
                    │ 3. ML classification                    │
                    │    soft-voting ensemble:                │
                    │      · XGBoost over numeric features    │
                    │      · SGD over char n-grams            │
                    │    loaded once at startup               │
                    └─────────────────────┬───────────────────┘
                                          ▼
                    ┌─────────────────────────────────────────┐
                    │ 4. Risk interpretation                  │
                    │    P(phishing) → verdict + 0-100 score  │
                    │    boundary = tuned operating point     │
                    └─────────────────────┬───────────────────┘
                                          ▼
                    ┌─────────────────────────────────────────┐
                    │ 5. Evidence collection                  │
                    │    observed indicators, each with the   │
                    │    literal substring that triggered it  │
                    └─────────────────────┬───────────────────┘
                                          ▼
                    ┌─────────────────────────────────────────┐
                    │ 6. AI explanation (LangChain)           │
                    │    verdict + evidence → LLM → prose     │
                    │    ✗ unavailable → rule-based fallback  │
                    │      (labelled as such, never faked)    │
                    └─────────────────────┬───────────────────┘
                                          ▼
                          ┌──────────────────────────────┐
                          │  Structured security report  │
                          └───────────────┬──────────────┘
                                          ▼
                          ┌──────────────────────────────┐
                          │  React dashboard             │
                          └──────────────────────────────┘
```

Nothing is written to a database at any step. The service is stateless; the
submitted URL exists only for the lifetime of the request.

---

## Why the LLM is not the classifier

| | ML classifier | LLM |
| --- | --- | --- |
| Latency | ~20 ms | 1–5 s |
| Cost per scan | ~0 | one API call |
| Determinism | Yes | No |
| Measurable precision/recall | Yes | Not meaningfully |
| Auditable decision | Yes (feature importances, threshold) | No |
| Works offline / without a key | Yes | No |

The classifier decides; the LLM explains. Step 6 runs strictly *after* step 3
and cannot change the verdict — it receives the finished result as an input.

---

## Preventing train/serve skew

The feature vector is the interface between training and inference. If the two
sides ever computed it differently, predictions would be silently wrong while
every test still passed.

PhishGuard has exactly one implementation, in `ml/features.py`:

```
ml/features.py  ──┬──► ml/preprocess.py       (training)
                  └──► backend/app/services/  (serving)
                         feature_extractor.py
```

The backend adds the project root to `sys.path` and imports `ml.features`
directly; the Docker image copies `ml/` beside `backend/` for the same reason.

Three additional guards:

1. `models/feature_metadata.json` records the exact ordered feature list. On
   startup the predictor compares it with `FEATURE_NAMES` and **refuses to
   serve** on a mismatch, rather than silently misaligning columns.
2. `backend/tests/test_predictor.py` asserts the serving frame's columns are
   `[url, *FEATURE_NAMES]`, in that order, and that the URL arrives
   canonicalised.
3. A dedicated CI job asserts the feature contract on every push.

---

## The two model branches

Both members of the ensemble receive the same input frame; each one's
`ColumnTransformer` decides what it actually consumes.

```
   input frame: [ url (canonical) | 52 numeric features ]
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
  ┌──────────────┐             ┌───────────────────┐
  │ XGBoost      │             │ SGD (log loss)    │
  │ numeric only │             │ numeric + char    │
  │              │             │ n-grams (3–5)     │
  └──────┬───────┘             └─────────┬─────────┘
         │                               │
         └───────────► soft vote ◄───────┘
                          │
                     P(phishing)
```

They capture different things and fail on different URLs:

- **Engineered numeric features** capture *structure*: entropy, subdomain
  depth, character ratios, lure-vocabulary counts, TLD reputation.
- **Character n-grams** capture *lexical* signal no counter enumerates:
  `-verify-`, `webscr`, `.com-`, brand misspellings.

Averaging their probabilities beat either alone on the validation split, which
is why the ensemble was selected — see `models/metrics.json` for the full
comparison across all five candidates.

Domain-grouped splitting is what makes the n-gram branch trustworthy: it cannot
score well by memorising domain strings, because every host in validation and
test is one it has never seen.

---

## Security posture

| Risk | Mitigation |
| --- | --- |
| SSRF via a submitted URL | The backend never issues a request to the submitted URL. There is no HTTP client in the analysis path at all. |
| Drive-by content execution | No page is fetched, rendered or parsed. Only the URL string is read. |
| Scheme-based injection (`javascript:`, `data:`) | Rejected at validation with an explicit error before any processing. |
| Resource exhaustion | URL length capped (2048), batch size capped (25), request timeouts configured. |
| Stack-trace disclosure | A catch-all handler returns an incident id; the traceback is logged server-side only. |
| Credential leakage through logs | `redact_url()` strips sensitive query parameters before any URL is logged. |
| Secret exposure to the browser | No secret is ever sent to the frontend. Only `VITE_API_URL` is public, by design. |
| Data retention | Nothing is stored. No database, no scan table, no user records. |

---

## Deployment: local vs. Azure

Local development requires nothing but Python and Node. The same codebase maps
onto Azure without modification:

```
   React bundle ──► Azure Static Web Apps  (or Blob + CDN)
                          │
                          ▼
   FastAPI ─────────► Azure Container Apps  (image from backend/Dockerfile)
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
   Model artifact                 Azure OpenAI
   (Azure Blob / Azure ML         (set LLM_PROVIDER=azure and the
    registry, mounted at           three AZURE_OPENAI_* variables —
    /app/models)                   no code change)
```

The provider switch is a configuration change only: `llm_analyzer.py` already
builds either `ChatOpenAI` or `AzureChatOpenAI` from environment variables.
Secrets belong in Azure Key Vault, surfaced as container environment
variables — the application never reads a credential from anywhere else.

Azure is entirely optional. Nothing in the local path depends on it.
