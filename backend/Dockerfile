# =============================================================================
# PhishGuard AI — API image
#
# Built from the repository root so that both `backend/` and `ml/` are in
# context: the API imports `ml.features` directly rather than duplicating the
# feature code, which is what guarantees there is no train/serve skew.
#
#   docker build -f backend/Dockerfile -t phishguard-api .
# =============================================================================

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 is required by XGBoost's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so the layer caches across source changes.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Application source.
COPY ml/ ./ml/
COPY backend/ ./backend/
COPY pyproject.toml ./

# Model artifacts. The build succeeds without them: the API starts in a
# degraded state and /api/health reports model_loaded=false, so an image can
# be built before the training pipeline has ever been run.
COPY models*/ ./models/

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 phishguard \
    && chown -R phishguard:phishguard /app
USER phishguard

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
