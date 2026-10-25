"""
app/main.py
────────────
FastAPI application factory for VADER.

Responsibilities:
  - Create and configure the FastAPI instance.
  - Register CORS middleware with environment-driven allowed origins.
  - Register global exception handlers for consistent error responses.
  - Mount all API routers under the /api prefix.
  - Provide the application lifespan context (startup / shutdown logging).

Security notes:
  - CORS allowed origins are read from config — never from user input.
  - All unhandled exceptions return a generic 500 body (no stack traces in prod).
  - Request bodies are size-limited by uvicorn / OS defaults; Pydantic schema
    limits enforce field-level constraints.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.config import get_settings
from app.models.schemas import ErrorDetail
from app.routers import baseline as baseline_router
from app.routers import health as health_router
from app.routers import query as query_router

logger = logging.getLogger(__name__)


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.

    Runs startup logic before yielding, then shutdown logic after.
    """
    settings = get_settings()
    settings.configure_logging()

    logger.info(
        "VADER starting | env=%s provider=%s debug=%s",
        settings.app_env.value,
        settings.llm_provider.value,
        settings.app_debug,
    )

    if not settings.active_key_present():
        logger.warning(
            "LLM provider '%s' is active but its API key is not set.  "
            "Multi-agent endpoints will fail until the key is configured.  "
            "Use LLM_PROVIDER=mock for keyless local development.",
            settings.llm_provider.value,
        )

    yield  # application is running

    logger.info("VADER shutting down.")


# ─── Application factory ──────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="VADER — Hybrid Multi-Agent AI System",
        description=(
            "Adversarial Collaborative Reasoning: multiple LLM agents debate, "
            "critique and synthesise answers to user queries."
        ),
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
        # Disable debug mode in production to prevent stack-trace leakage
        debug=settings.app_debug and settings.app_env.value != "production",
    )

    _register_cors(app, settings)
    _register_exception_handlers(app, settings)
    _register_routers(app)

    return app


# ─── CORS ─────────────────────────────────────────────────────────────────────

def _register_cors(app: FastAPI, settings) -> None:
    """
    Configure CORS middleware.

    Origins are read from the CORS_ALLOWED_ORIGINS environment variable.
    In production, this must be an explicit list — never '*'.
    """
    origins = settings.cors_origins_list

    if not origins:
        logger.warning("No CORS origins configured.  Frontend requests will be blocked.")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "Accept"],
    )
    logger.debug("CORS configured | origins=%s", origins)


# ─── Exception handlers ───────────────────────────────────────────────────────

def _register_exception_handlers(app: FastAPI, settings) -> None:
    """Register centralised exception handlers for consistent error responses."""

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """
        Handle Pydantic / FastAPI request validation failures (422 → 400).

        Returns the standard ErrorDetail schema so the frontend always gets a
        consistent shape.  Raw Pydantic error details are included in the
        'details' field for debugging.
        """
        logger.info(
            "Request validation error | path=%s errors=%s",
            request.url.path,
            exc.errors(),
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorDetail(
                error="validation_error",
                message="One or more request fields failed validation.",
                details={"errors": exc.errors()},
            ).model_dump(mode="json"),
        )

    @app.exception_handler(ValidationError)
    async def _handle_pydantic_error(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        """Handle standalone Pydantic validation errors (from model instantiation)."""
        logger.warning(
            "Pydantic validation error | path=%s", request.url.path
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorDetail(
                error="schema_error",
                message="Response schema validation failed.",
                details={"errors": exc.errors()},
            ).model_dump(mode="json"),
        )

    @app.exception_handler(ValueError)
    async def _handle_value_error(
        request: Request, exc: ValueError
    ) -> JSONResponse:
        """Handle ValueError raised by application logic (e.g. unknown LLM provider)."""
        logger.warning(
            "ValueError | path=%s msg=%s", request.url.path, str(exc)
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorDetail(
                error="invalid_input",
                message=str(exc),
            ).model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def _handle_generic_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Catch-all handler.

        Stack traces are logged server-side but NEVER returned to the client in
        production to prevent information leakage.
        """
        logger.error(
            "Unhandled exception | path=%s type=%s",
            request.url.path,
            type(exc).__name__,
            exc_info=True,
        )
        # In debug mode return the exception type for easier local development
        detail = (
            f"{type(exc).__name__}: {exc}"
            if settings.app_debug
            else "An internal server error occurred."
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorDetail(
                error="internal_error",
                message=detail,
            ).model_dump(mode="json"),
        )


# ─── Routers ──────────────────────────────────────────────────────────────────

def _register_routers(app: FastAPI) -> None:
    """Mount all API routers."""
    app.include_router(health_router.router, prefix="/api")
    app.include_router(query_router.router, prefix="/api")
    app.include_router(baseline_router.router, prefix="/api")


# ─── Application instance ─────────────────────────────────────────────────────

# This module-level instance is what uvicorn imports.
app = create_app()
