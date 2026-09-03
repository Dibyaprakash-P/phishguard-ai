"""Find the accuracy-optimal decision threshold for the deployed model.

The shipped threshold is tuned to maximise recall subject to a precision floor
(default 0.97), which deliberately trades accuracy away to avoid false alarms
on legitimate sites. This script reports what the *same already-trained model*
would score at other thresholds, so the trade-off is measurable rather than
assumed.

It does not retrain anything, and it modifies nothing unless ``--apply`` is
passed. It only scores the existing model over the split it was evaluated on.

Method
------
1. Rebuild the identical train/validation/test split (same seed, same row cap,
   same domain-grouped strategy) that ``ml/train.py`` used. The row cap is read
   from the model metadata rather than guessed - see the warning below.
2. Score the deployed ``models/phishing_model.joblib`` on validation and test.
3. Choose the accuracy-maximising threshold on **validation only**, then report
   it on test. Selecting the threshold on the split you report would inflate
   the number.

Why the row cap matters
-----------------------
``subsample`` draws a class-balanced sample before the split. A different cap
draws a *different* sample, so domains the model was trained on can land in the
rebuilt "test" set. That is exactly the leakage this project exists to avoid,
and it silently inflates every metric - in one measured case ROC-AUC went from
0.959 to 0.988. The cap therefore defaults to the value recorded in
``models/feature_metadata.json``, and overriding it prints a loud warning.

Usage
-----
    python scripts/tune_threshold.py            # reproduces the deployed split
    python scripts/tune_threshold.py --apply    # adopt the accuracy-optimal cut

Runtime is a couple of minutes: it re-reads ~1.45M raw rows and extracts
features for the sampled corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml import config  # noqa: E402
from ml.dataset_loaders import load_training_corpora  # noqa: E402
from ml.preprocess import (  # noqa: E402
    build_model_frame,
    clean_corpus,
    group_split,
    subsample,
)


def scores(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """Metric suite at one decision threshold."""
    predicted = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
    }


def line(label: str, s: dict) -> None:
    print(
        f"  {label:34s} thr={s['threshold']:.4f}  acc={s['accuracy']:.4f}  "
        f"prec={s['precision']:.4f}  rec={s['recall']:.4f}  f1={s['f1']:.4f}"
    )


def resolve_row_cap(args: argparse.Namespace, metadata: dict) -> int | None:
    """Return the row cap that reproduces the deployed split, or None on error."""
    trained_cap = metadata.get("dataset", {}).get("max_rows_cap")

    if args.max_rows is None:
        if trained_cap is None:
            print("error: the model metadata records no row cap; pass --max-rows.")
            return None
        print(f"Using the row cap recorded in the model metadata: {int(trained_cap):,}")
        return int(trained_cap)

    if trained_cap is not None and int(args.max_rows) != int(trained_cap):
        print(
            f"WARNING: --max-rows {args.max_rows:,} does not match the cap the model "
            f"was trained with ({int(trained_cap):,})."
        )
        print("         The rebuilt split is NOT the held-out one: its test rows may")
        print("         contain domains the model trained on, which inflates every")
        print("         metric below. Do not quote these numbers.")
        print()
    return int(args.max_rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report the accuracy-optimal threshold for the deployed model"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help=(
            "Row cap. Defaults to the cap recorded in the model metadata, which is "
            "the only value that reproduces the split the model was evaluated on."
        ),
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the accuracy-optimal threshold into models/feature_metadata.json.",
    )
    args = parser.parse_args()

    if not config.MODEL_PATH.is_file():
        print(f"error: no model at {config.MODEL_PATH}. Run: python -m ml.train")
        return 1

    metadata = json.loads(config.FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    max_rows = resolve_row_cap(args, metadata)
    if max_rows is None:
        return 1

    started = time.perf_counter()

    print("Rebuilding the deployed split (this reads the full corpus)...")
    cleaned, _ = clean_corpus(load_training_corpora())
    sampled = subsample(cleaned, max_rows, args.seed)
    _, val, test = group_split(sampled, config.TEST_SIZE, config.VALIDATION_SIZE, args.seed)
    print(f"  validation={len(val):,} rows   test={len(test):,} rows")

    recorded_test = metadata.get("dataset", {}).get("test_rows")
    if recorded_test is not None and int(recorded_test) != len(test):
        print(
            f"  WARNING: rebuilt test split has {len(test):,} rows but the model "
            f"metadata records {int(recorded_test):,}."
        )
        print("           The split does not match; the numbers below are not the")
        print("           held-out ones. Do not quote them.")
    print()

    print("Loading the deployed model and scoring...")
    model = joblib.load(config.MODEL_PATH)
    shipped = float(metadata.get("decision_threshold", 0.5))

    p_val = model.predict_proba(build_model_frame(val["url"]))[:, 1]
    p_test = model.predict_proba(build_model_frame(test["url"]))[:, 1]
    y_val, y_test = val["label"].to_numpy(), test["label"].to_numpy()

    # Select on validation, report on test. Never select on the split you report.
    grid = np.arange(0.05, 0.96, 0.005)
    best = float(max(grid, key=lambda t: accuracy_score(y_val, (p_val >= t).astype(int))))

    print()
    print(f"Model: {metadata.get('model_name')} v{metadata.get('model_version')}")
    print(
        f"Test ROC-AUC: {roc_auc_score(y_test, p_test):.4f}  "
        "(threshold-independent, unchanged by any tuning below)"
    )
    print()

    print("TEST SPLIT:")
    line("shipped (precision floor)", scores(y_test, p_test, shipped))
    line("accuracy-optimal", scores(y_test, p_test, best))

    print()
    print("Accuracy across the threshold range (test):")
    for t in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, shipped, 0.70, 0.80, 0.90):
        s = scores(y_test, p_test, t)
        marker = "  <- shipped" if abs(t - shipped) < 1e-9 else ""
        print(
            f"  thr={t:.4f}  acc={s['accuracy']:.4f}  prec={s['precision']:.4f}  "
            f"rec={s['recall']:.4f}{marker}"
        )

    ceiling = max(accuracy_score(y_test, (p_test >= t).astype(int)) for t in grid)
    print()
    print(f"Ceiling: best test accuracy at ANY threshold = {ceiling:.4f}")
    print("Thresholding only slides along a fixed curve. Beating this needs better")
    print("features, more data, or a different task definition - not a new cut-off.")

    if args.apply:
        metadata["decision_threshold"] = best
        metadata["threshold_policy"] = "maximise accuracy on the validation split"
        config.FEATURE_METADATA_PATH.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print()
        print(f"Wrote threshold {best:.4f} to {config.FEATURE_METADATA_PATH}")
        print("Restart the API for it to take effect.")
    else:
        print()
        print("(Nothing was modified. Re-run with --apply to adopt the new threshold.)")

    print()
    print(f"Done in {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
