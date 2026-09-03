"""Liveness / readiness endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.core.config import settings
from backend.app.schemas.analysis import HealthResponse, LLMStatus
from backend.app.services.llm_analyzer import llm_analyzer
from backend.app.services.predictor import predictor

router = APIRouter(prefix="/api", tags=["system"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health",
    description=(
        "Reports whether the ML model artifact loaded and whether an LLM provider is "
        "configured. The service reports 'degraded' rather than failing when the model "
        "is missing, so the UI can tell the user to run the training pipeline."
    ),
)
async def health() -> HealthResponse:
    """Return the current operational state of the API."""
    return HealthResponse(
        status="ok" if predictor.is_loaded else "degraded",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        model_loaded=predictor.is_loaded,
        model_name=predictor.model_name if predictor.is_loaded else None,
        llm=LLMStatus(
            configured=llm_analyzer.available,
            provider=settings.llm_provider,
            model=llm_analyzer.model_label,
        ),
        agent_enabled=settings.enable_agent,
    )
