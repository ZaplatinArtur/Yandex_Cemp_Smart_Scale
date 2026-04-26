from __future__ import annotations

from fastapi import Request

from smart_scale.api.errors import ServiceUnavailableError
from smart_scale.ml.pipeline import RecognitionPipeline


def get_pipeline(request: Request) -> RecognitionPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise ServiceUnavailableError("Пайплайн инференса не инициализирован.")
    return pipeline
