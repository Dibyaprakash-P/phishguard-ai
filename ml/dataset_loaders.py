"""Loaders that normalise each bundled corpus into a common schema.

Every loader returns a :class:`pandas.DataFrame` with exactly three columns:

``url``    - the raw URL string as published by the source
``label``  - 0 = legitimate, 1 = phishing/malicious
``source`` - short identifier of the originating corpus

Each loader is defensive: it inspects the columns actually present in the file
before reading it and raises a clear error instead of silently assuming a
schema. Missing files are skipped with a warning rather than aborting the run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml.config import RAW_DATA_DIRS

logger = logging.getLogger(__name__)

COLUMNS = ["url", "label", "source"]


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def find_dataset(filename: str) -> Path | None:
    """Locate ``filename`` in any of the configured raw-data directories."""
    for directory in RAW_DATA_DIRS:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def peek_columns(path: Path) -> list[str]:
    """Read only the header row so schema assumptions can be validated."""
    return list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)


def _require(path: Path, needed: list[str]) -> None:
    """Fail loudly when a source file does not have the expected columns."""
    present = peek_columns(path)
    missing = [c for c in needed if c not in present]
    if missing:
        raise ValueError(
            f"{path.name}: expected column(s) {missing} but found {present[:12]}. "
            "Update ml/dataset_loaders.py if the upstream schema changed."
        )


def _frame(urls: pd.Series, labels: pd.Series, source: str) -> pd.DataFrame:
    frame = pd.DataFrame({"url": urls.astype(str), "label": labels.astype(int)})
    frame["source"] = source
    return frame[COLUMNS]


# --------------------------------------------------------------------------
# Per-corpus loaders
# --------------------------------------------------------------------------

def load_phiusiil(path: Path) -> pd.DataFrame:
    """UCI PhiUSIIL Phishing URL Dataset.

    Upstream encodes ``label = 1`` as *legitimate*; PhishGuard uses the
    opposite convention, so the label is inverted here. Only the ``URL`` and
    ``label`` columns are used - the dataset's own page-content features
    (line-of-code counts, favicon presence, ...) are deliberately discarded
    because they require visiting the site, which this project never does.
    """
    _require(path, ["URL", "label"])
    raw = pd.read_csv(path, usecols=["URL", "label"], encoding="utf-8-sig")
    return _frame(raw["URL"], 1 - raw["label"], "phiusiil")


def load_malicious_phish(path: Path) -> pd.DataFrame:
    """Kaggle "Malicious URLs" dataset (url, type).

    ``type`` is multi-class. ``benign`` maps to 0; ``phishing``, ``malware``
    and ``defacement`` all map to 1 because PhishGuard's product question is
    "is this link safe to click", not "which malware family is it".
    """
    _require(path, ["url", "type"])
    raw = pd.read_csv(path, usecols=["url", "type"])
    raw["type"] = raw["type"].astype(str).str.strip().str.lower()
    known = {"benign", "phishing", "malware", "defacement"}
    unknown = set(raw["type"].unique()) - known
    if unknown:
        raise ValueError(f"{path.name}: unexpected type values {sorted(unknown)}")
    return _frame(raw["url"], (raw["type"] != "benign").astype(int), "malicious_phish")


def load_phishing_site_urls(path: Path) -> pd.DataFrame:
    """Kaggle "Phishing Site URLs" dataset (URL, Label) with good/bad labels."""
    _require(path, ["URL", "Label"])
    raw = pd.read_csv(path, usecols=["URL", "Label"])
    raw["Label"] = raw["Label"].astype(str).str.strip().str.lower()
    unknown = set(raw["Label"].unique()) - {"good", "bad"}
    if unknown:
        raise ValueError(f"{path.name}: unexpected Label values {sorted(unknown)}")
    return _frame(raw["URL"], (raw["Label"] == "bad").astype(int), "phishing_site_urls")


def load_dataset_phishing(path: Path) -> pd.DataFrame:
    """Hannousse & Yahiouche balanced corpus (url + 87 features + status).

    Only ``url`` and ``status`` are consumed; the bundled features are ignored
    in favour of PhishGuard's own extractor so that training and inference use
    identical code.
    """
    _require(path, ["url", "status"])
    raw = pd.read_csv(path, usecols=["url", "status"])
    raw["status"] = raw["status"].astype(str).str.strip().str.lower()
    unknown = set(raw["status"].unique()) - {"legitimate", "phishing"}
    if unknown:
        raise ValueError(f"{path.name}: unexpected status values {sorted(unknown)}")
    return _frame(raw["url"], (raw["status"] == "phishing").astype(int), "dataset_phishing")


def load_phishtank(path: Path) -> pd.DataFrame:
    """PhishTank ``verified_online.csv`` - verified phishing URLs only.

    This corpus contains **no** legitimate examples, so it is never mixed into
    training. It is held out entirely and used as an independent, real-world
    recall check - see ``external_holdout_check`` in ``ml/train.py``.
    """
    _require(path, ["url"])
    raw = pd.read_csv(path, usecols=["url"])
    return _frame(raw["url"], pd.Series(1, index=raw.index), "phishtank")


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """Description of one supported corpus."""

    key: str
    filename: str
    loader: Callable[[Path], pd.DataFrame]
    description: str
    training: bool = True


TRAINING_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "phiusiil",
        "PhiUSIIL_Phishing_URL_Dataset.csv",
        load_phiusiil,
        "UCI PhiUSIIL Phishing URL Dataset (235k URLs, label inverted)",
    ),
    DatasetSpec(
        "malicious_phish",
        "malicious_phish.csv",
        load_malicious_phish,
        "Kaggle Malicious URLs (651k URLs, 4 classes collapsed to binary)",
    ),
    DatasetSpec(
        "phishing_site_urls",
        "phishing_site_urls.csv",
        load_phishing_site_urls,
        "Kaggle Phishing Site URLs (549k URLs, good/bad)",
    ),
    DatasetSpec(
        "dataset_phishing",
        "dataset_phishing.csv",
        load_dataset_phishing,
        "Hannousse & Yahiouche balanced web-page phishing corpus (11k URLs)",
    ),
)

HOLDOUT_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "phishtank",
        "verified_online.csv",
        load_phishtank,
        "PhishTank verified online phishing feed (external recall holdout)",
        training=False,
    ),
)

ALL_DATASETS: tuple[DatasetSpec, ...] = TRAINING_DATASETS + HOLDOUT_DATASETS


def load_spec(spec: DatasetSpec) -> pd.DataFrame | None:
    """Load one corpus, returning ``None`` when its file is absent."""
    path = find_dataset(spec.filename)
    if path is None:
        logger.warning(
            "Dataset %s not found (expected %s in %s) - skipping.",
            spec.key,
            spec.filename,
            " or ".join(str(d) for d in RAW_DATA_DIRS),
        )
        return None
    logger.info("Loading %s from %s", spec.key, path)
    frame = spec.loader(path)
    logger.info(
        "  %s: %d rows (phishing=%d, legitimate=%d)",
        spec.key,
        len(frame),
        int((frame["label"] == 1).sum()),
        int((frame["label"] == 0).sum()),
    )
    return frame


def load_training_corpora() -> pd.DataFrame:
    """Concatenate every available training corpus."""
    frames = [f for f in (load_spec(s) for s in TRAINING_DATASETS) if f is not None]
    if not frames:
        raise FileNotFoundError(
            "No training datasets found. See data/README.md for download "
            "instructions and place the CSV files in ./datasets/."
        )
    return pd.concat(frames, ignore_index=True)


def load_holdout_corpora() -> pd.DataFrame:
    """Concatenate the external holdout corpora (may be empty)."""
    frames = [f for f in (load_spec(s) for s in HOLDOUT_DATASETS) if f is not None]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
