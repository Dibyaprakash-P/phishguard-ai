"""PhishGuard AI - FastAPI application entry point.

Run locally:
    uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.core.exceptions import register_exception_handlers
from backend.app.core.logging_config import configure_logging
from backend.app.routes import analyze, health, model
from backend.app.services.predictor import predictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model artifact once, at startup.

    Inference must never touch disk or retrain, so everything expensive happens
    here. A missing artifact degrades the service instead of killing it: the
    API stays up and ``/api/health`` reports ``model_loaded: false``.
    """
    configure_logging()
    logger.info("Starting %s v%s (%s)", settings.app_name, settings.app_version,
                settings.environment)

    if predictor.load():
        logger.info("Model ready: %s v%s (threshold=%.4f)",
                    predictor.model_name, predictor.model_version, predictor.threshold)
    else:
        logger.warning("Starting WITHOUT a model. Analysis requests will return 503 "
                       "until you run: python -m ml.train")

    logger.info(
        "LLM provider=%s configured=%s | LangGraph agent=%s",
        settings.llm_provider, settings.llm_configured, settings.enable_agent,
    )

    yield

    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title="PhishGuard AI",
    version=settings.app_version,
    description=(
        "AI-powered phishing URL detection.\n\n"
        "A gradient-boosted classifier scores URLs from statically extracted "
        "lexical and structural features; an LLM then explains that verdict via "
        "LangChain. **The analyzer never visits, resolves or fetches the submitted "
        "URL**, and no submitted URL is persisted - the service is stateless and "
        "has no database."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(model.router)


@app.get("/", tags=["system"], summary="API root")
async def root() -> dict:
    """Point callers at the docs and the main endpoints."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "endpoints": ["/api/health", "/api/analyze", "/api/batch-analyze", "/api/model-info"],
        "static_analysis_only": True,
    }
