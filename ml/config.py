"""Central configuration for the PhishGuard AI training pipeline."""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

#: Where the raw, user-supplied corpora live. ``datasets/`` is the folder the
#: downloaded CSV/ARFF files are dropped into; ``data/raw`` is also scanned so
#: either layout works.
RAW_DATA_DIRS: tuple[Path, ...] = (
    PROJECT_ROOT / "datasets",
    PROJECT_ROOT / "data" / "raw",
)

PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"

MODEL_PATH: Path = MODELS_DIR / "phishing_model.joblib"
FEATURE_METADATA_PATH: Path = MODELS_DIR / "feature_metadata.json"
METRICS_PATH: Path = MODELS_DIR / "metrics.json"

#: Cached, deduplicated (url, label, domain) corpus produced by preprocess.py.
CORPUS_PATH: Path = PROCESSED_DIR / "corpus.parquet"
CORPUS_CSV_PATH: Path = PROCESSED_DIR / "corpus.csv"

#: Local MLflow tracking store (file-backed - no server or database required).
MLFLOW_TRACKING_URI: str = (PROJECT_ROOT / "mlruns").as_uri()
MLFLOW_EXPERIMENT: str = "phishguard-url-detection"

# --------------------------------------------------------------------------
# Split / training configuration
# --------------------------------------------------------------------------

RANDOM_SEED: int = 42
TEST_SIZE: float = 0.15
VALIDATION_SIZE: float = 0.15

#: Default cap on training rows. The combined corpus is ~1.4M URLs; the cap
#: keeps a full pipeline run to a few minutes on a laptop. Override with
#: ``--max-rows 0`` to train on everything.
DEFAULT_MAX_ROWS: int = 400_000

LABEL_LEGITIMATE: int = 0
LABEL_PHISHING: int = 1

# --------------------------------------------------------------------------
# Risk scoring (kept here so training and serving share the same thresholds)
# --------------------------------------------------------------------------

#: Probability thresholds mapping model output -> UI verdict.
#: These are product decisions, not universal scientific constants.
#:
#: These values must stay equal to their counterparts in
#: ``backend/app/core/config.py``, which carries the measurements behind each
#: number. They had silently drifted (0.40 here, 0.48 there), so the sanity
#: gate was scoring a band the API does not serve; ``backend/tests/
#: test_pipeline.py`` now asserts they agree.
SUSPICIOUS_THRESHOLD: float = 0.48
PHISHING_THRESHOLD: float = 0.70

#: Ceiling applied to P(phishing) for URLs on the curated known-good list in
#: ``ml/reputation.py``. See the matching setting in the backend config for
#: the rationale and the measurements.
REPUTATION_CEILING: float = 0.25

#: Risk-score bands (0-100).
RISK_LOW_MAX: int = 30
RISK_MEDIUM_MAX: int = 70
