"""Domain exceptions and the handlers that turn them into safe API responses.

Rule: clients receive a stable ``error`` code plus a human-readable message.
Stack traces and internal details are logged server-side only.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class PhishGuardError(Exception):
    """Base class for expected, client-presentable failures."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "internal_error"

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class InvalidURLError(PhishGuardError):
    """The submitted string is not a URL this analyzer can process."""

    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "invalid_url"


class ModelNotAvailableError(PhishGuardError):
    """No trained model artifact could be loaded."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "model_unavailable"


class PredictionError(PhishGuardError):
    """Inference failed after the model had been loaded."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "prediction_failed"


class BatchTooLargeError(PhishGuardError):
    """More URLs were submitted than the configured batch limit allows."""

    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    error_code = "batch_too_large"


class AgentUnavailableError(PhishGuardError):
    """The optional LangGraph agent was requested but is not enabled."""

    status_code = status.HTTP_501_NOT_IMPLEMENTED
    error_code = "agent_unavailable"


def _payload(error: str, message: str, detail: str | None = None) -> dict:
    body = {"error": error, "message": message}
    if detail:
        body["detail"] = detail
    return body


def register_exception_handlers(app: FastAPI) -> None:
    """Attach JSON error handlers to ``app``."""

    @app.exception_handler(PhishGuardError)
    async def _handle_known(_: Request, exc: PhishGuardError) -> JSONResponse:
        logger.warning("%s: %s", exc.error_code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(exc.error_code, exc.message, exc.detail),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        message = first.get("msg", "Request validation failed")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_payload(
                "validation_error",
                f"{field}: {message}".strip(": ") if field else message,
            ),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Never leak the traceback: log it with a correlation id and hand the
        # client only that id.
        incident = uuid.uuid4().hex[:12]
        logger.exception("Unhandled exception [incident=%s]", incident)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_payload(
                "internal_error",
                "An unexpected error occurred while processing the request.",
                f"incident={incident}",
            ),
        )
