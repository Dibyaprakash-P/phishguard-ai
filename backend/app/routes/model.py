"""Model metadata and measured-performance endpoint.

Every number served here is read from the artifacts written by
``ml/train.py``. Nothing is hardcoded: if the pipeline has not been run, the
endpoint reports ``trained: false`` and the UI says so, rather than displaying
invented metrics.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from backend.app.schemas.analysis import (
    ExternalHoldout,
    ModelComparisonEntry,
    ModelInfoResponse,
    ModelMetrics,
    OperatingPoint,
    SanityCheck,
)
from backend.app.services.predictor import predictor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["model"])

_METRIC_FIELDS = (
    "accuracy", "precision", "recall", "f1",
    "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate",
)


def _to_metrics(raw: dict[str, Any] | None) -> ModelMetrics | None:
    """Coerce a metrics dict from disk, returning None when incomplete."""
    if not raw:
        return None
    try:
        return ModelMetrics(**{field: float(raw[field]) for field in _METRIC_FIELDS})
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Metrics payload incomplete or malformed: %s", exc)
        return None


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model metadata and measured metrics",
    description=(
        "Returns the trained model's identity, the leakage-controlled evaluation "
        "metrics measured on the held-out test split, the full model comparison and "
        "the external holdout result. Values come from models/metrics.json."
    ),
)
async def model_info() -> ModelInfoResponse:
    """Serve everything the Model Intelligence page renders."""
    if not predictor.is_loaded:
        return ModelInfoResponse(
            trained=False,
            message=(
                predictor.load_error
                or "Model not trained. Run the training pipeline: python -m ml.train"
            ),
        )

    metadata = predictor.metadata
    metrics = predictor.metrics

    comparison = []
    for entry in metrics.get("comparison", []):
        validation = entry.get("validation", {})
        try:
            comparison.append(ModelComparisonEntry(
                model=entry["model"],
                accuracy=float(validation["accuracy"]),
                precision=float(validation["precision"]),
                recall=float(validation["recall"]),
                f1=float(validation["f1"]),
                roc_auc=float(validation["roc_auc"]),
                pr_auc=float(validation["pr_auc"]),
                train_seconds=entry.get("train_seconds"),
            ))
        except (KeyError, TypeError, ValueError):
            logger.warning("Skipping malformed comparison entry: %s", entry.get("model"))

    holdout_raw = metrics.get("external_holdout")
    holdout = None
    if holdout_raw:
        try:
            holdout = ExternalHoldout(**{
                k: holdout_raw[k]
                for k in ("source", "rows", "unique_domains", "detected", "recall", "note")
            })
        except (KeyError, TypeError, ValueError):
            logger.warning("Skipping malformed external holdout payload")

    sanity = None
    if metrics.get("sanity_check"):
        try:
            sanity = SanityCheck(**metrics["sanity_check"])
        except (TypeError, ValueError):
            logger.warning("Skipping malformed sanity-check payload")

    operating_points = []
    for point in metrics.get("operating_points", []):
        try:
            operating_points.append(OperatingPoint(**point))
        except (TypeError, ValueError):
            logger.warning("Skipping malformed operating point: %s", point)

    return ModelInfoResponse(
        trained=True,
        model_name=metadata.get("model_name"),
        model_version=metadata.get("model_version"),
        trained_at=metadata.get("trained_at"),
        feature_count=metadata.get("feature_count"),
        feature_names=list(metadata.get("feature_names", [])),
        decision_threshold=metadata.get("decision_threshold"),
        threshold_policy=metadata.get("threshold_policy"),
        selection_criterion=metrics.get("selection_criterion"),
        test_metrics=_to_metrics(metrics.get("test")),
        validation_metrics=_to_metrics(metrics.get("validation")),
        comparison=comparison,
        external_holdout=holdout,
        sanity_check=sanity,
        operating_points=operating_points,
        precision_floor=metrics.get("precision_floor"),
        top_features=metadata.get("top_features", []),
        dataset=metadata.get("dataset", {}),
        risk_bands=metadata.get("risk_bands", {}),
    )
