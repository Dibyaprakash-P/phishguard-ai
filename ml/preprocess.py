"""Corpus assembly, cleaning and leakage-aware splitting.

Pipeline
--------
1. Load and normalise every available corpus (``ml.dataset_loaders``).
2. Drop empty / malformed / unparseable URLs.
3. Deduplicate on the *normalised* URL string.
4. Resolve label conflicts (the same URL labelled both ways across sources).
5. Attach the registrable domain to every row.
6. Split train/validation/test **by domain group**, never by row.

Why group splitting matters
---------------------------
Public phishing feeds contain many URLs per host
(``evil.tk/login``, ``evil.tk/verify``, ``evil.tk/confirm``...). A random row
split would put near-identical siblings on both sides of the split, so the
model would be scored on hosts it had already memorised. Splitting on the
registrable domain guarantees that every host in the test set is one the model
has never seen, which is the only setting that reflects production use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml import config
from ml.dataset_loaders import load_holdout_corpora, load_training_corpora
from ml.features import (
    FEATURE_NAMES,
    canonicalize_for_features,
    extract_features,
    normalize_url,
    registrable_domain,
    split_url,
)

logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    """Feature matrices and label vectors for each split."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    stats: dict[str, object]


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def clean_corpus(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean and deduplicate a raw ``(url, label, source)`` frame."""
    report: dict[str, int] = {"input_rows": len(frame)}

    frame = frame.copy()
    frame["url"] = frame["url"].astype(str).str.strip()

    # 1. Structurally impossible rows.
    frame = frame[frame["url"].str.len() > 3]
    frame = frame[~frame["url"].str.lower().isin({"nan", "none", "null", "url"})]
    report["after_length_filter"] = len(frame)

    # 2. Normalise, then require a parseable host with at least one dot
    #    (or a literal IP). Anything else cannot be a real navigable URL.
    frame["normalized"] = frame["url"].map(normalize_url)
    hosts = frame["normalized"].map(lambda u: split_url(u)["host"])
    frame["host"] = hosts
    valid_host = hosts.str.contains(".", regex=False) & (hosts.str.len() >= 4)
    frame = frame[valid_host]
    report["after_host_filter"] = len(frame)

    # 3. Conflicting labels: the same URL appearing as both legitimate and
    #    phishing across sources is unusable supervision - drop every copy.
    #    This MUST run before deduplication, otherwise `keep="first"` would
    #    silently resolve conflicts by source ordering instead of surfacing
    #    them, and the disagreement would be baked into the labels.
    conflicts = frame.groupby("normalized")["label"].transform("nunique")
    conflicting = int((conflicts > 1).sum())
    if conflicting:
        frame = frame[conflicts == 1]
    report["label_conflicts_removed"] = conflicting
    report["after_conflict_filter"] = len(frame)

    # 4. Deduplicate on the normalised form so that "example.com/x" and
    #    "http://example.com/x" are recognised as the same record.
    frame = frame.drop_duplicates(subset=["normalized"], keep="first")
    report["after_dedup"] = len(frame)
    report["final_rows"] = len(frame)

    frame["domain"] = frame["host"].map(registrable_domain)
    return frame.reset_index(drop=True), report


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def subsample(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Class-balanced subsample capped at ``max_rows`` (0 = keep everything).

    Sampling is stratified by label so the cap never skews the prior, and it is
    applied *before* the split so all three splits shrink proportionally.
    """
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    per_class = max_rows // 2
    parts = []
    for _label, group in frame.groupby("label"):
        take = min(per_class, len(group))
        parts.append(group.sample(n=take, random_state=seed))
    sampled = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    logger.info("Subsampled %d -> %d rows (cap=%d)", len(frame), len(sampled), max_rows)
    return sampled


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def group_split(
    frame: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by registrable domain so no domain spans two splits."""
    rng = np.random.default_rng(seed)
    domains = frame["domain"].unique()
    rng.shuffle(domains)

    n_domains = len(domains)
    n_test = int(n_domains * test_size)
    n_val = int(n_domains * val_size)

    test_domains = set(domains[:n_test])
    val_domains = set(domains[n_test:n_test + n_val])

    in_test = frame["domain"].isin(test_domains)
    in_val = frame["domain"].isin(val_domains)

    train = frame[~(in_test | in_val)]
    val = frame[in_val]
    test = frame[in_test]

    # Contract check: an empty intersection is the whole point of this function.
    assert not (set(train["domain"]) & set(test["domain"])), "domain leakage: train/test"
    assert not (set(train["domain"]) & set(val["domain"])), "domain leakage: train/val"

    logger.info(
        "Split by domain -> train=%d val=%d test=%d rows (%d unique domains)",
        len(train), len(val), len(test), n_domains,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# --------------------------------------------------------------------------
# Featurisation
# --------------------------------------------------------------------------

#: Name of the column carrying the canonicalised URL string. The hybrid model
#: consumes it with a character n-gram vectorizer alongside the numeric
#: features; numeric-only candidates drop it via their ColumnTransformer.
MODEL_URL_COLUMN = "url"


def build_feature_matrix(urls: pd.Series) -> pd.DataFrame:
    """Apply the shared extractor to a URL series -> numeric feature DataFrame."""
    records = [extract_features(u) for u in urls]
    matrix = pd.DataFrame.from_records(records, columns=list(FEATURE_NAMES))
    return matrix.astype("float32")


def build_model_frame(urls: pd.Series) -> pd.DataFrame:
    """Build the full model input: canonical URL text plus numeric features.

    Every candidate model accepts this one frame, so training and serving share
    a single input contract regardless of which model was selected.
    """
    matrix = build_feature_matrix(urls)
    matrix.insert(0, MODEL_URL_COLUMN, [canonicalize_for_features(u) for u in urls])
    return matrix


def prepare_dataset(
    max_rows: int = config.DEFAULT_MAX_ROWS,
    seed: int = config.RANDOM_SEED,
    cache: bool = True,
) -> SplitData:
    """Run the full preprocessing pipeline and return ready-to-train splits."""
    raw = load_training_corpora()
    cleaned, report = clean_corpus(raw)

    logger.info("Cleaning report: %s", report)
    source_counts = cleaned.groupby(["source", "label"]).size().unstack(fill_value=0)
    logger.info("Rows per source after cleaning:\n%s", source_counts)

    if cache:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cleaned[["url", "label", "source", "domain"]].to_csv(
            config.CORPUS_CSV_PATH, index=False
        )
        logger.info("Cached cleaned corpus -> %s", config.CORPUS_CSV_PATH)

    sampled = subsample(cleaned, max_rows, seed)
    train, val, test = group_split(sampled, config.TEST_SIZE, config.VALIDATION_SIZE, seed)

    logger.info("Extracting features for %d URLs...", len(sampled))
    stats: dict[str, object] = {
        "cleaning_report": report,
        "rows_per_source": {
            str(k): {str(kk): int(vv) for kk, vv in v.items()}
            for k, v in source_counts.to_dict("index").items()
        },
        "sampled_rows": int(len(sampled)),
        "max_rows_cap": max_rows,
        "unique_domains": int(sampled["domain"].nunique()),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_phishing_rate": float(train["label"].mean()),
        "val_phishing_rate": float(val["label"].mean()),
        "test_phishing_rate": float(test["label"].mean()),
        "feature_count": len(FEATURE_NAMES),
        "split_strategy": "GroupShuffle by registrable domain (no domain in two splits)",
    }

    return SplitData(
        X_train=build_model_frame(train["url"]),
        y_train=train["label"].reset_index(drop=True),
        X_val=build_model_frame(val["url"]),
        y_val=val["label"].reset_index(drop=True),
        X_test=build_model_frame(test["url"]),
        y_test=test["label"].reset_index(drop=True),
        stats=stats,
    )


def prepare_holdout(train_domains: set[str] | None = None) -> pd.DataFrame:
    """Load the external PhishTank holdout, optionally excluding seen domains."""
    holdout = load_holdout_corpora()
    if holdout.empty:
        return holdout
    cleaned, _ = clean_corpus(holdout)
    if train_domains:
        before = len(cleaned)
        cleaned = cleaned[~cleaned["domain"].isin(train_domains)]
        logger.info(
            "External holdout: dropped %d rows on domains seen in training",
            before - len(cleaned),
        )
    return cleaned.reset_index(drop=True)


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = prepare_dataset()
    print(data.stats)
