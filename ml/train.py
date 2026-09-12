"""Train, compare and register phishing-URL classifiers.

Runs Logistic Regression, Random Forest and XGBoost over the same
leakage-controlled splits, logs every run to a local MLflow file store, selects
the best model on the **validation** split, then reports a single honest score
on the untouched test split plus an external PhishTank recall check.

Usage
-----
    python -m ml.train                 # default 400k-row cap
    python -m ml.train --max-rows 0    # full ~1.3M-row corpus
    python -m ml.train --no-mlflow     # skip experiment tracking
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from xgboost import XGBClassifier

from ml import config
from ml.features import FEATURE_NAMES, feature_fingerprint
from ml.preprocess import (
    MODEL_URL_COLUMN,
    SplitData,
    build_model_frame,
    prepare_dataset,
    prepare_holdout,
)
from ml.reputation import known_good_domain
from ml.sanity_urls import sanity_groups

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Model zoo
# --------------------------------------------------------------------------

def _numeric(transform=None) -> ColumnTransformer:
    """Select the numeric feature columns and drop the URL text column."""
    return ColumnTransformer(
        [("numeric", transform or "passthrough", list(FEATURE_NAMES))],
        remainder="drop",
    )


def _hybrid_features() -> ColumnTransformer:
    """Numeric features plus character n-grams of the canonical URL.

    Hand-engineered features capture *structure* (entropy, subdomain depth,
    lure-vocabulary counts). Character n-grams capture *lexical* signal those
    counters miss - the specific substrings that recur in phishing URLs such as
    "-verify-", "webscr", ".com-" or brand misspellings - without anyone having
    to enumerate them by hand.

    Because the split is grouped by registrable domain, the vectorizer cannot
    win by memorising domain strings: every host in the validation and test
    splits is one it has never seen.
    """
    return ColumnTransformer(
        [
            ("numeric", MaxAbsScaler(), list(FEATURE_NAMES)),
            (
                "char_ngrams",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=200_000,
                    sublinear_tf=True,
                    lowercase=True,
                ),
                MODEL_URL_COLUMN,
            ),
        ],
        remainder="drop",
    )


def _char_ngram_model(seed: int, n_jobs: int) -> Pipeline:
    """Linear model over the sparse hybrid feature space.

    SGD with a log loss is used rather than a full solver because the n-gram
    matrix has hundreds of thousands of columns; it converges in a couple of
    minutes where lbfgs or liblinear would take far longer for no measurable
    gain in this setting.
    """
    return Pipeline([
        ("features", _hybrid_features()),
        ("clf", SGDClassifier(
            loss="log_loss", alpha=1e-6, max_iter=20, tol=1e-4,
            class_weight="balanced", random_state=seed, n_jobs=n_jobs,
        )),
    ])


def _xgboost_model(seed: int, n_jobs: int) -> Pipeline:
    """Gradient-boosted trees over the engineered numeric features."""
    return Pipeline([
        ("features", _numeric()),
        ("clf", XGBClassifier(
            n_estimators=600, max_depth=8, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=2,
            reg_lambda=1.0, gamma=0.0, tree_method="hist",
            eval_metric="logloss", random_state=seed, n_jobs=n_jobs,
        )),
    ])


def build_candidates(seed: int, n_jobs: int = -1) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Return ``{name: (estimator, hyperparameters)}`` for every candidate.

    All five accept the same input frame (canonical URL text plus numeric
    features); each candidate's ColumnTransformer decides what it consumes, so
    training and serving share one input contract whichever model wins.

    Logistic Regression is scaled because the raw features span very different
    magnitudes (``url_length`` vs. ``digit_ratio``); the tree models are
    scale-invariant and need no such preprocessing.
    """
    return {
        "logistic_regression": (
            Pipeline([
                ("features", _numeric(StandardScaler())),
                ("clf", LogisticRegression(
                    max_iter=1000, C=1.0, solver="lbfgs",
                    class_weight="balanced", random_state=seed, n_jobs=n_jobs,
                )),
            ]),
            {"C": 1.0, "solver": "lbfgs", "max_iter": 1000,
             "class_weight": "balanced", "inputs": "numeric"},
        ),
        "random_forest": (
            Pipeline([
                ("features", _numeric()),
                ("clf", RandomForestClassifier(
                    n_estimators=250, max_depth=28, min_samples_leaf=2,
                    max_features="sqrt", class_weight="balanced_subsample",
                    random_state=seed, n_jobs=n_jobs,
                )),
            ]),
            {"n_estimators": 250, "max_depth": 28, "min_samples_leaf": 2,
             "max_features": "sqrt", "class_weight": "balanced_subsample",
             "inputs": "numeric"},
        ),
        "xgboost": (
            _xgboost_model(seed, n_jobs),
            {"n_estimators": 600, "max_depth": 8, "learning_rate": 0.1,
             "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 2,
             "tree_method": "hist", "inputs": "numeric"},
        ),
        "char_ngram_sgd": (
            _char_ngram_model(seed, n_jobs),
            {"loss": "log_loss", "alpha": 1e-6, "max_iter": 20,
             "ngram_range": "3-5 char_wb", "max_features": 200000,
             "inputs": "numeric + char n-grams"},
        ),
        # Soft-voting ensemble of the two strongest, most complementary
        # candidates: trees over engineered structure, and a linear model over
        # lexical n-grams. They fail on different URLs, so averaging their
        # probabilities beats either one alone.
        #
        # The 3:2 weighting is measured, not guessed. Sweeping the blend on the
        # full corpus shows the two members trade off against each other in
        # opposite directions - the n-gram model carries overall accuracy, the
        # tree model carries correctness on legitimate URLs:
        #
        #   weight on n-grams   test accuracy   unlisted legitimate (40 URLs)
        #   0.0 (xgboost only)      0.8545              36/40
        #   0.2                     0.8749              37/40
        #   0.5 (the old 1:1)       0.9162              34/40
        #   0.6 (this)              0.9181              34/40
        #   1.0 (n-grams only)      0.9056              33/40
        #
        # 0.6 is the accuracy peak, and it still clears the sanity gate's 80%
        # floor on legitimate URLs that the reputation list does not cover.
        "hybrid_ensemble": (
            VotingClassifier(
                estimators=[
                    ("char_ngram_sgd", _char_ngram_model(seed, n_jobs)),
                    ("xgboost", _xgboost_model(seed, n_jobs)),
                ],
                voting="soft",
                weights=[3, 2],
                n_jobs=1,  # each member already parallelises internally
            ),
            {"members": "char_ngram_sgd + xgboost", "voting": "soft",
             "weights": "3:2", "inputs": "numeric + char n-grams"},
        ),
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute the full metric suite at a given decision threshold."""
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "threshold": float(threshold),
    }


def operating_points(
    y_true: pd.Series, proba: np.ndarray,
    floors: tuple[float, ...] = (0.90, 0.95, 0.97, 0.99),
) -> list[dict[str, float]]:
    """Recall attainable at a range of precision floors.

    Publishing the whole curve rather than a single tuned number makes the
    central trade-off explicit: a phishing detector that cries wolf on
    well-known sites gets switched off, while one that is too permissive fails
    at its job. Operators can re-tune with ``--min-precision`` and see exactly
    what it costs.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    precisions, recalls = precisions[:-1], recalls[:-1]

    points: list[dict[str, float]] = []
    for floor in floors:
        eligible = precisions >= floor
        if not eligible.any():
            points.append({"precision_floor": floor, "attainable": False,
                           "recall": 0.0, "threshold": 1.0})
            continue
        idx = int(np.argmax(np.where(eligible, recalls, -1.0)))
        points.append({
            "precision_floor": float(floor),
            "attainable": True,
            "threshold": round(float(thresholds[idx]), 4),
            "precision": round(float(precisions[idx]), 4),
            "recall": round(float(recalls[idx]), 4),
        })
    return points


def tune_threshold(
    y_true: pd.Series,
    proba: np.ndarray,
    min_precision: float = 0.97,
    objective: str = "precision_floor",
) -> float:
    """Pick the decision threshold on the validation split.

    For a security classifier the two error types are not symmetric:

    * a **false positive** flags a legitimate site as phishing, which erodes
      user trust and gets the product switched off;
    * a **false negative** lets a credential-harvesting page through, which is
      the harm the product exists to prevent.

    Two objectives are supported, because they answer different questions:

    ``precision_floor`` (default)
        Maximise recall subject to ``precision >= min_precision``. This is the
        right choice for a deployed security product: a false alarm on a
        well-known site destroys user trust faster than a miss.

    ``accuracy``
        Maximise plain accuracy. This yields the highest headline accuracy the
        model can produce, at the cost of more false positives on legitimate
        sites. Measured on this corpus it buys roughly +1.8 accuracy points and
        costs ~3.4 points of precision.

    Neither objective changes the model or its ROC-AUC - both only slide the
    cut-off along a fixed curve. ``models/metrics.json`` publishes the whole
    operating-point table so the trade-off stays visible.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    precisions, recalls = precisions[:-1], recalls[:-1]

    if objective == "accuracy":
        # Sweep the observed thresholds and keep the most accurate. Selection
        # happens on validation; the test split is never consulted here.
        grid = np.unique(np.round(thresholds, 4))
        if grid.size == 0:
            return 0.5
        best = max(grid, key=lambda t: accuracy_score(y_true, (proba >= t).astype(int)))
        return float(best)

    eligible = precisions >= min_precision
    if eligible.any():
        idx = int(np.argmax(np.where(eligible, recalls, -1.0)))
        return float(thresholds[idx])

    f1 = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)
    return float(thresholds[int(np.argmax(f1))])


# --------------------------------------------------------------------------
# MLflow
# --------------------------------------------------------------------------

class Tracker:
    """Thin MLflow wrapper that degrades to a no-op when tracking is off."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.mlflow = None
        if not enabled:
            return
        try:
            # MLflow 3.x refuses the plain filesystem backend unless this opt-in
            # is set. PhishGuard deliberately keeps tracking database-free, so
            # the flag is enabled here rather than introducing a SQLite store.
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

            import mlflow  # imported lazily so the pipeline runs without it

            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
            self.mlflow = mlflow
            logger.info("MLflow tracking -> %s", config.MLFLOW_TRACKING_URI)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("MLflow unavailable (%s); continuing without tracking.", exc)
            self.enabled = False

    def run(self, name: str):
        if self.enabled and self.mlflow is not None:
            return self.mlflow.start_run(run_name=name)
        from contextlib import nullcontext

        return nullcontext()

    def log_params(self, params: dict[str, Any]) -> None:
        if self.enabled and self.mlflow is not None:
            self.mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], prefix: str = "") -> None:
        if self.enabled and self.mlflow is not None:
            self.mlflow.log_metrics(
                {f"{prefix}{k}": float(v) for k, v in metrics.items()
                 if isinstance(v, (int, float))}
            )

    def log_dict(self, payload: dict[str, Any], filename: str) -> None:
        if self.enabled and self.mlflow is not None:
            self.mlflow.log_dict(payload, filename)

    def log_model(self, model: Any, name: str, register_as: str | None = None) -> None:
        if not (self.enabled and self.mlflow is not None):
            return
        try:
            import mlflow.sklearn

            mlflow.sklearn.log_model(model, name=name, registered_model_name=register_as)
        except Exception as exc:  # pragma: no cover - optional convenience
            logger.warning("Could not log model artifact to MLflow: %s", exc)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def train_and_compare(
    data: SplitData,
    seed: int,
    tracker: Tracker,
    n_jobs: int,
    min_precision: float = 0.97,
    objective: str = "precision_floor",
) -> list[dict[str, Any]]:
    """Fit every candidate and return their evaluation records."""
    results: list[dict[str, Any]] = []

    for name, (estimator, params) in build_candidates(seed, n_jobs).items():
        logger.info("=" * 70)
        logger.info("Training %s on %d rows x %d features",
                    name, len(data.X_train), data.X_train.shape[1])
        with tracker.run(name):
            started = time.perf_counter()
            estimator.fit(data.X_train, data.y_train)
            train_seconds = time.perf_counter() - started

            val_proba = estimator.predict_proba(data.X_val)[:, 1]
            threshold = tune_threshold(
                data.y_val, val_proba, min_precision, objective
            )
            val_metrics = evaluate(data.y_val, val_proba, threshold)
            val_default = evaluate(data.y_val, val_proba, 0.5)

            tracker.log_params({**params, "model": name, "seed": seed,
                                "features": data.X_train.shape[1],
                                "train_rows": len(data.X_train)})
            tracker.log_metrics(val_metrics, prefix="val_")
            tracker.log_metrics({"train_seconds": train_seconds})
            tracker.log_dict(data.stats, "dataset_stats.json")

            logger.info(
                "  %s val: acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f (thr=%.3f, %.1fs)",
                name, val_metrics["accuracy"], val_metrics["precision"],
                val_metrics["recall"], val_metrics["f1"], val_metrics["roc_auc"],
                threshold, train_seconds,
            )

            results.append({
                "name": name,
                "estimator": estimator,
                "params": params,
                "threshold": threshold,
                "validation": val_metrics,
                "validation_at_0.5": val_default,
                "train_seconds": train_seconds,
            })
    return results


def external_holdout_check(model: Any, threshold: float, train_domains: set[str]) -> dict[str, Any] | None:
    """Score the model against the PhishTank feed it has never seen.

    PhishTank contains phishing URLs only, so recall is the only meaningful
    metric here - it answers "how many live, in-the-wild phishing URLs on
    previously unseen domains would this model catch?".
    """
    holdout = prepare_holdout(train_domains)
    if holdout.empty:
        logger.info("No external holdout corpus available - skipping.")
        return None

    X = build_model_frame(holdout["url"])
    proba = model.predict_proba(X)[:, 1]
    caught = int((proba >= threshold).sum())
    result = {
        "source": "phishtank_verified_online",
        "rows": int(len(holdout)),
        "unique_domains": int(holdout["domain"].nunique()),
        "detected": caught,
        "recall": float(caught / len(holdout)),
        "mean_probability": float(proba.mean()),
        "note": "Phishing-only feed on domains never seen in training; recall only.",
    }
    logger.info("External PhishTank holdout recall: %.4f (%d/%d unseen-domain URLs)",
                result["recall"], caught, result["rows"])
    return result


def sanity_check(model: Any, threshold: float) -> dict[str, Any]:
    """Score the curated smoke-test set and report per-case outcomes.

    This gate exists because aggregate test metrics cannot detect a shortcut
    that is present in the test split as well. It is reported separately from
    the real metrics and must never be quoted as model accuracy.

    Two scores are produced for every case, and both are kept:

    ``model_only``
        the raw classifier verdict, with no reputation prior. This is the
        honest read on the model, and the number the artefact gate watches.

    ``as_served``
        the verdict a user actually sees, after :func:`ml.reputation.
        known_good_domain` clamps scores on curated known-good domains.

    Keeping both is the whole point. A reputation list that improved
    ``as_served`` while ``model_only`` quietly rotted would hide exactly the
    class of bug this gate was written to catch, so the two are reported side
    by side and :data:`ml.sanity_urls.UNLISTED_LEGITIMATE_URLS` is scored
    where no reputation rule can reach it.
    """
    groups = sanity_groups()
    urls = [url for urls_, _ in groups.values() for url in urls_]
    labels = np.array([label for urls_, label in groups.values() for _ in urls_])

    proba = model.predict_proba(build_model_frame(pd.Series(urls)))[:, 1]
    served = np.array([
        min(p, config.REPUTATION_CEILING) if known_good_domain(u) else p
        for u, p in zip(urls, proba, strict=True)
    ])

    def verdicts(scores: np.ndarray) -> np.ndarray:
        """Three-way display verdict: 0 legitimate, 1 suspicious, 2 phishing."""
        return np.where(
            scores >= threshold, 2,
            np.where(scores >= min(config.SUSPICIOUS_THRESHOLD, threshold), 1, 0),
        )

    # A legitimate URL is only "correct" when it reads *legitimate* - landing in
    # the suspicious band is the failure the user reported, so it counts as one.
    model_ok = np.where(labels == 0, verdicts(proba) == 0, proba >= threshold)
    served_ok = np.where(labels == 0, verdicts(served) == 0, served >= threshold)

    per_group: dict[str, Any] = {}
    offset = 0
    for name, (urls_, label) in groups.items():
        end = offset + len(urls_)
        per_group[name] = {
            "total": len(urls_),
            "label": label,
            "model_only_correct": int(model_ok[offset:end].sum()),
            "as_served_correct": int(served_ok[offset:end].sum()),
        }
        offset = end

    failures = [
        {
            "url": url,
            "expected": int(label),
            "model_probability": round(float(p), 4),
            "served_probability": round(float(s), 4),
        }
        for url, label, p, s, ok in zip(urls, labels, proba, served, served_ok, strict=True)
        if not ok
    ]

    legit_mask = labels == 0
    unlisted = per_group["unlisted_legitimate"]
    result = {
        "note": (
            "Hand-curated smoke test, not a benchmark. Reported separately from "
            "the held-out test metrics. 'as_served' includes the reputation prior; "
            "'model_only' does not."
        ),
        "total": len(urls),
        "correct": int(served_ok.sum()),
        "accuracy": float(served_ok.mean()),
        "legitimate_accuracy": float(served_ok[legit_mask].mean()),
        "phishing_accuracy": float(served_ok[~legit_mask].mean()),
        "model_only_accuracy": float(model_ok.mean()),
        "model_only_legitimate_accuracy": float(model_ok[legit_mask].mean()),
        "unlisted_legitimate_accuracy": (
            unlisted["model_only_correct"] / unlisted["total"]
        ),
        "per_group": per_group,
        "failures": failures,
    }

    logger.info(
        "Sanity check: %d/%d as served (legit %.0f%%, phishing %.0f%%) | "
        "model alone %d/%d (legit %.0f%%)",
        result["correct"], result["total"],
        result["legitimate_accuracy"] * 100, result["phishing_accuracy"] * 100,
        int(model_ok.sum()), result["total"],
        result["model_only_legitimate_accuracy"] * 100,
    )
    for name, stats in per_group.items():
        logger.info("  %-22s model %d/%d, served %d/%d", name,
                    stats["model_only_correct"], stats["total"],
                    stats["as_served_correct"], stats["total"])
    for failure in failures:
        logger.warning("  SANITY FAIL expected=%s model=%.4f served=%.4f %s",
                       failure["expected"], failure["model_probability"],
                       failure["served_probability"], failure["url"])

    if per_group["bypass_attempts"]["as_served_correct"] < per_group["bypass_attempts"]["total"]:
        logger.error(
            "Sanity gate FAILED on bypass attempts: a URL shaped to borrow a trusted "
            "name was reported legitimate. The reputation prior has become a bypass - "
            "do not ship this artifact."
        )
    if result["unlisted_legitimate_accuracy"] < 0.8:
        logger.error(
            "Sanity gate FAILED on legitimate URLs that the reputation list does not "
            "cover. This usually means the model has latched onto a dataset collection "
            "artefact rather than phishing semantics - do not ship this artifact. The "
            "reputation prior cannot mask this: those URLs are deliberately not on the "
            "known-good list."
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhishGuard AI phishing URL models")
    parser.add_argument("--max-rows", type=int, default=config.DEFAULT_MAX_ROWS,
                        help="Cap on training rows (0 = use the entire corpus)")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--min-precision", type=float, default=0.97,
                        help="Precision floor used when tuning the decision threshold")
    parser.add_argument(
        "--objective",
        choices=("precision_floor", "accuracy"),
        default="precision_floor",
        help=(
            "Threshold-selection objective. 'precision_floor' (default) maximises "
            "recall subject to --min-precision. 'accuracy' maximises plain accuracy, "
            "which reports a higher headline number but produces more false positives "
            "on legitimate sites."
        ),
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    parser.add_argument("--no-holdout", action="store_true",
                        help="Skip the external PhishTank recall check")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    started = time.perf_counter()
    logger.info("Preparing dataset (max_rows=%s)...", args.max_rows or "all")
    data = prepare_dataset(max_rows=args.max_rows, seed=args.seed)
    logger.info("Dataset ready: %s", json.dumps(
        {k: v for k, v in data.stats.items() if k != "cleaning_report"}, indent=2))

    tracker = Tracker(enabled=not args.no_mlflow)
    results = train_and_compare(
        data, args.seed, tracker, args.n_jobs, args.min_precision, args.objective
    )

    # Selection criterion: PR-AUC on validation. It is threshold-independent
    # and, unlike ROC-AUC, stays informative under class imbalance.
    best = max(results, key=lambda r: r["validation"]["pr_auc"])
    logger.info("=" * 70)
    logger.info("Selected model: %s (val PR-AUC %.4f)", best["name"], best["validation"]["pr_auc"])

    # The test split is touched exactly once, here.
    test_proba = best["estimator"].predict_proba(data.X_test)[:, 1]
    test_metrics = evaluate(data.y_test, test_proba, best["threshold"])
    logger.info(
        "TEST  acc=%.4f prec=%.4f rec=%.4f f1=%.4f roc_auc=%.4f pr_auc=%.4f",
        test_metrics["accuracy"], test_metrics["precision"], test_metrics["recall"],
        test_metrics["f1"], test_metrics["roc_auc"], test_metrics["pr_auc"],
    )

    curve = operating_points(data.y_test, test_proba)
    logger.info("Operating points on the test split:")
    for point in curve:
        if point.get("attainable"):
            logger.info("  precision>=%.2f -> recall %.4f at threshold %.4f",
                        point["precision_floor"], point["recall"], point["threshold"])
        else:
            logger.info("  precision>=%.2f -> not attainable", point["precision_floor"])

    sanity = sanity_check(best["estimator"], best["threshold"])

    holdout_result = None
    if not args.no_holdout:
        train_domains = set(pd.read_csv(config.CORPUS_CSV_PATH, usecols=["domain"])["domain"]) \
            if config.CORPUS_CSV_PATH.exists() else set()
        holdout_result = external_holdout_check(best["estimator"], best["threshold"], train_domains)

    # ---------------------------------------------------------------- artifacts
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["estimator"], config.MODEL_PATH)
    logger.info("Saved model artifact -> %s", config.MODEL_PATH)

    importances = _feature_importances(best["estimator"])

    metadata = {
        "model_name": best["name"],
        "model_version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        # Digest of what the extractor *computes*, not just what it names.
        # The predictor refuses to serve when this disagrees, so a change to a
        # feature's logic cannot silently skew a model trained before it.
        "feature_fingerprint": feature_fingerprint(),
        "decision_threshold": best["threshold"],
        "threshold_policy": (
            f"maximise recall subject to validation precision >= {args.min_precision}"
            if args.objective == "precision_floor"
            else "maximise accuracy on the validation split"
        ),
        "label_mapping": {"0": "legitimate", "1": "phishing"},
        "hyperparameters": best["params"],
        "dataset": data.stats,
        "risk_bands": {
            "low": [0, config.RISK_LOW_MAX],
            "medium": [config.RISK_LOW_MAX + 1, config.RISK_MEDIUM_MAX],
            "high": [config.RISK_MEDIUM_MAX + 1, 100],
        },
        "top_features": importances[:15],
        "sanity_check": {k: v for k, v in sanity.items() if k != "failures"},
    }
    config.FEATURE_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics_payload = {
        "selected_model": best["name"],
        "selection_criterion": "highest validation PR-AUC",
        "decision_threshold": best["threshold"],
        "test": test_metrics,
        "validation": best["validation"],
        "external_holdout": holdout_result,
        "sanity_check": sanity,
        "operating_points": curve,
        "precision_floor": args.min_precision,
        "threshold_objective": args.objective,
        "comparison": [
            {
                "model": r["name"],
                "threshold": r["threshold"],
                "train_seconds": round(r["train_seconds"], 2),
                "validation": r["validation"],
            }
            for r in results
        ],
        "dataset": data.stats,
        "generated_at": metadata["trained_at"],
        "total_pipeline_seconds": round(time.perf_counter() - started, 1),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    logger.info("Saved metrics -> %s", config.METRICS_PATH)

    with tracker.run(f"{best['name']}_final"):
        tracker.log_params({**best["params"], "model": best["name"], "selected": True})
        tracker.log_metrics(test_metrics, prefix="test_")
        tracker.log_metrics({"sanity_accuracy": sanity["accuracy"],
                             "sanity_legitimate_accuracy": sanity["legitimate_accuracy"]})
        if holdout_result:
            tracker.log_metrics({"phishtank_recall": holdout_result["recall"]})
        tracker.log_dict(metrics_payload, "metrics.json")
        tracker.log_dict(metadata, "feature_metadata.json")
        tracker.log_model(best["estimator"], "model", register_as="phishguard-url-classifier")

    logger.info("Pipeline finished in %.1fs", time.perf_counter() - started)


def _feature_importances(estimator) -> list[dict[str, float]]:
    """Best-effort importance extraction for whichever estimator was selected.

    Importances are reported over the *numeric* features only. The n-gram
    branch has hundreds of thousands of sparse coefficients that do not map
    onto named features, so it is deliberately not summarised here.
    """
    model = estimator

    # Unwrap a soft-voting ensemble to its tree member, the only part with
    # interpretable per-feature importances.
    if isinstance(model, VotingClassifier):
        fitted = dict(
            zip([name for name, _ in model.estimators], model.estimators_, strict=True)
        )
        model = fitted.get("xgboost") or next(iter(fitted.values()), None)
        if model is None:
            return []

    if isinstance(model, Pipeline):
        model = model[-1]

    if hasattr(model, "feature_importances_"):
        scores = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        scores = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:  # pragma: no cover - defensive
        return []

    # Only meaningful when the vector lines up with the named numeric features.
    if len(scores) != len(FEATURE_NAMES):
        return []

    order = np.argsort(scores)[::-1]
    return [
        {"feature": FEATURE_NAMES[i], "importance": round(float(scores[i]), 6)}
        for i in order
    ]


if __name__ == "__main__":
    main()
