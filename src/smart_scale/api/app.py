from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from smart_scale.api.errors import ServiceUnavailableError
from smart_scale.api.routes import router
from smart_scale.config import Settings, get_settings
from smart_scale.ml.pipeline import RecognitionPipeline


LOGGER = logging.getLogger("smart_scale.api")


def create_app(
    settings: Settings | None = None,
    pipeline_factory: Callable[[Settings], RecognitionPipeline] | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_factory = pipeline_factory or RecognitionPipeline.from_settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _configure_logging()
        start_time = time.perf_counter()
        LOGGER.info("event=api_startup_begin vector_backend=%s", resolved_settings.vector_backend)

        try:
            pipeline = resolved_factory(resolved_settings)
            pipeline.warmup()
        except Exception:
            LOGGER.exception("event=api_startup_failed")
            raise

        app.state.pipeline = pipeline
        app.state.settings = resolved_settings

        health = pipeline.health_status()
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        LOGGER.info(
            "event=api_startup_completed duration_ms=%s warmup_completed=%s catalog_items=%s vector_backend=%s",
            duration_ms,
            health.get("warmup_completed"),
            health.get("catalog_items"),
            health.get("vector_backend"),
        )

        try:
            yield
        finally:
            close_method = getattr(pipeline, "close", None)
            if callable(close_method):
                close_method()
            LOGGER.info("event=api_shutdown_completed")

    app = FastAPI(
        title=resolved_settings.api_title,
        version="0.1.0",
        summary="Inference API for the smart scale recognition pipeline.",
        lifespan=lifespan,
    )
    app.include_router(router, prefix="/api")
    _register_exception_handlers(app)

    @app.get("/")
    async def root() -> dict[str, str]:
        return {
            "service": resolved_settings.api_title,
            "docs": "/docs",
            "health": "/api/health",
            "predict": "/api/predict",
        }

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceUnavailableError)
    async def handle_service_unavailable(_request: Request, exc: ServiceUnavailableError) -> JSONResponse:
        LOGGER.warning("event=service_unavailable detail=%s", exc.detail)
        return JSONResponse(status_code=503, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("event=unhandled_exception path=%s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка сервиса."})


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )


app = create_app()
