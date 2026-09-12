# Datasets

PhishGuard AI trains on **public phishing URL corpora**. This document records
exactly which files are expected, where they come from, what their columns look
like, and how they are combined.

Nothing here is downloaded automatically. The training pipeline works entirely
offline once the files are in place, and it *never* fetches a URL from any
dataset — every feature is computed from the URL string.

---

## Where to put the files

Drop the raw files into either directory; both are scanned:

```
datasets/          <- default location (already used in this repo)
data/raw/          <- alternative
```

Only the files that are present are used. A missing dataset is logged and
skipped, so the pipeline still runs with a subset.

### Fetching them

Two of the four need no account:

```bash
mkdir -p datasets

# PhiUSIIL (UCI) - ~235k rows
curl -sSL -o datasets/phiusiil.zip "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
python -c "import zipfile; zipfile.ZipFile('datasets/phiusiil.zip').extractall('datasets')"
rm datasets/phiusiil.zip

# Hannousse & Yahiouche (Mendeley) - 11,430 rows, balanced
curl -sSL -o datasets/dataset_phishing.csv "https://data.mendeley.com/public-files/datasets/c2gw7fy2j4/files/575316f4-ee1d-453e-a04f-7b950915b61b/file_downloaded"
```

The two Kaggle corpora need an account. Put a `kaggle.json` API token in
`~/.kaggle/` and run:

```bash
pip install kaggle
kaggle datasets download -d sid321axn/malicious-urls-dataset -p datasets --unzip
kaggle datasets download -d taruntiwarihp/phishing-site-urls -p datasets --unzip
```

**`malicious_phish.csv` is not optional in practice.** See the warning below.

---

## Expected files

### 1. PhiUSIIL Phishing URL Dataset — `PhiUSIIL_Phishing_URL_Dataset.csv`

- **Source:** UCI Machine Learning Repository, dataset 967
  <https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset>
- **Size:** ~235,000 rows
- **Columns used:** `URL`, `label`
- **⚠ Label convention:** upstream uses `1 = legitimate`, `0 = phishing` —
  the **opposite** of PhishGuard's convention. `ml/dataset_loaders.py`
  inverts it on load.
- **Note:** the dataset also ships ~50 page-content features (line counts,
  favicon presence, iframe counts…). These are **deliberately discarded**:
  obtaining them requires visiting the site, which this project never does.
  Only the raw URL is used.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

### 2. Malicious URLs dataset — `malicious_phish.csv`

- **Source:** Kaggle, "Malicious URLs dataset" (sid321axn)
  <https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset>
- **Size:** ~651,000 rows
- **Columns used:** `url`, `type`
- **Label mapping:** `benign → 0`; `phishing`, `malware`, `defacement → 1`.
  The multi-class labels are collapsed because the product question is
  "is this link safe to click", not "which malware family is it".

### 3. Phishing Site URLs — `phishing_site_urls.csv`

- **Source:** Kaggle, "Phishing Site URLs" (taruntiwarihp)
  <https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls>
- **Size:** ~549,000 rows
- **Columns used:** `URL`, `Label` (`good` / `bad`)
- **Label mapping:** `bad → 1`, `good → 0`
- **Observed overlap:** almost all of its `good` rows are duplicates of rows in
  `malicious_phish.csv`, so after deduplication this corpus contributes
  essentially only phishing examples. This is expected, and is reported in the
  cleaning summary printed by the pipeline.

### 4. Web page phishing detection — `dataset_phishing.csv`

- **Source:** Hannousse & Yahiouche, "Web page phishing detection" (Mendeley Data)
  <https://data.mendeley.com/datasets/c2gw7fy2j4>
- **Size:** 11,430 rows (balanced 50/50)
- **Columns used:** `url`, `status` (`legitimate` / `phishing`)
- **Note:** its 87 precomputed features are ignored in favour of PhishGuard's
  own extractor, so that training and inference run identical code.

### 5. PhishTank verified feed — `verified_online.csv`  *(holdout only)*

- **Source:** PhishTank <https://phishtank.org/developer_info.php>
- **Size:** ~74,000 rows
- **Columns used:** `url`
- **Never used for training.** This feed contains phishing URLs only, and is
  far more recent than the other corpora. It is held out entirely and used as
  an **independent real-world recall check**: rows on any registrable domain
  seen during training are removed first, so the score reflects generalisation
  to genuinely unseen hosts.

---

## Label convention

PhishGuard normalises everything to:

| label | meaning                               |
| ----- | ------------------------------------- |
| `0`   | legitimate                            |
| `1`   | phishing / malicious                  |

---

## How the corpora are combined

`ml/preprocess.py` runs the following, in order:

1. **Load and normalise.** Each loader validates the columns it expects before
   reading and raises a clear error if the upstream schema has changed. No
   column is ever assumed.
2. **Drop unusable rows** — empty, placeholder (`nan`, `null`), or with no
   parseable host.
3. **Resolve label conflicts.** A URL that appears as legitimate in one corpus
   and phishing in another is unusable supervision, so **every copy is
   dropped**. This runs *before* deduplication — otherwise `keep="first"`
   would silently resolve the disagreement by source ordering.
4. **Deduplicate** on the normalised URL, so `example.com/x` and
   `http://example.com/x` count once.
5. **Attach the registrable domain** to each row.
6. **Class-balanced subsample** to the configured row cap.
7. **Split by domain group** — see below.

Typical cleaning summary for the four training corpora:

```
input rows                1,447,762
after host filter         1,447,503
label conflicts removed     195,922   <- the corpora disagree on ~14% of overlapping URLs
after deduplication         851,279
unique registrable domains  320,876
```

---

## Preventing data leakage

Two forms of leakage are guarded against explicitly.

### Domain-level leakage

Public phishing feeds contain many URLs per host (`evil.tk/login`,
`evil.tk/verify`, `evil.tk/confirm`…). A random row split would place
near-identical siblings on both sides, and the model would be scored on hosts
it had already memorised.

PhishGuard therefore splits on the **registrable domain**: every host in the
validation and test splits is one the model has never seen. `ml/preprocess.py`
asserts the split intersections are empty, and `backend/tests/test_pipeline.py`
tests it.

### Why `malicious_phish.csv` is effectively required

Measured over the corpora as loaded, on the canonical URL form:

| source | label | no path | has path |
| --- | --- | --- | --- |
| PhiUSIIL | legitimate | **100.00%** | 0.00% |
| PhiUSIIL | phishing | 40.14% | 59.86% |
| dataset_phishing | legitimate | 6.16% | 93.84% |
| dataset_phishing | phishing | 5.02% | 94.98% |

Every single legitimate URL in PhiUSIIL is path-less. Train on PhiUSIIL as the
bulk of the corpus and "has a path" becomes a near-free phishing signal worth
60% of its positive class — which reports as *high accuracy* while making the
model worse at the thing it exists to do. `malicious_phish.csv` contributes
~380k benign URLs that all have paths, and it is what breaks that correlation.

`dataset_phishing.csv` is balanced on this axis but is only 11k rows, far too
small to counterweight 236k on its own.

If you train without `malicious_phish.csv`, treat the resulting accuracy as
unvalidated and watch the `unlisted_legitimate` sanity group, which is where
this artefact shows up first.

This lowers the reported scores considerably compared with a random split —
that is the point. The numbers mean something.

### Provenance leakage (the collection artefact)

Measured over the assembled corpus:

| source (label 0)  | starts `www.` | uses `https` | has a path |
| ----------------- | ------------- | ------------ | ---------- |
| PhiUSIIL          | **100 %**     | **100 %**    | **0 %**    |
| malicious_phish   | 0.04 %        | 0.5 %        | **100 %**  |

Those columns describe *how each dataset was collected*, not whether a URL is a
phishing page. A model given them learns to recognise the source corpus: an
early build of this project scored 92% test accuracy while confidently
classifying `https://google.com` as phishing — a bare, https, path-less URL
matches neither legitimate style.

The fix is in `ml.features.canonicalize_for_features`: the scheme and a leading
`www.` are stripped before extraction, and `has_https` / `has_www_prefix` were
removed as model inputs. HTTPS is still shown to the user for transparency; it
is simply not something the model is allowed to rely on. (It is a weak signal
regardless — certificates are free and most live phishing serves TLS.)

`ml/sanity_urls.py` now runs after every training run as a permanent gate
against this class of bug.

---

## Generated files

The pipeline writes:

```
data/processed/corpus.csv     cleaned (url, label, source, domain)  — git-ignored
models/phishing_model.joblib  the selected model artifact
models/feature_metadata.json  feature contract, threshold, dataset provenance
models/metrics.json           measured metrics, model comparison, holdout, sanity
mlruns/                       local MLflow file store — git-ignored
```

Raw corpora and generated data are **git-ignored**: several are tens of
megabytes and they are all publicly redistributable from their original
sources. Only the model artifacts are small enough to be worth committing, and
that is a per-project decision (Git LFS is a reasonable alternative).

---

## Refreshing the data

The PhishTank feed changes daily; the Kaggle and UCI corpora are static
snapshots. Phishing vocabulary and hosting patterns drift, so a model trained
today will degrade. Re-download the feeds and re-run the pipeline periodically:

```bash
python -m ml.train
```

Compare the new run against previous ones in the MLflow UI before promoting it.
