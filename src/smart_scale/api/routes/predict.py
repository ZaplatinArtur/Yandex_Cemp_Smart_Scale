from __future__ import annotations

import io
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from smart_scale.api.dependencies import get_pipeline
from smart_scale.api.errors import ServiceUnavailableError
from smart_scale.api.schemas import CropResponse, HealthResponse, MatchResponse, PredictionResponse
from smart_scale.ml.pipeline import RecognitionPipeline


router = APIRouter(tags=["smart-scale"])
LOGGER = logging.getLogger("smart_scale.api.predict")


@router.post("/predict", response_model=PredictionResponse)
async def predict_product(
    request: Request,
    image: UploadFile = File(...),
    weight_grams: float = Form(...),
    top_k: int = Form(3, ge=1),
    pipeline: RecognitionPipeline = Depends(get_pipeline),
) -> PredictionResponse:
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Файл изображения пустой.")

    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Ожидается файл изображения.")

    try:
        pil_image = Image.open(io.BytesIO(payload)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Не удалось распознать изображение.") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Не удалось обработать изображение.") from exc

    try:
        result = await run_in_threadpool(pipeline.run, pil_image, weight_grams, top_k)
    except (FileNotFoundError, ImportError, OSError, RuntimeError) as exc:
        raise ServiceUnavailableError("Пайплайн инференса временно недоступен.") from exc

    LOGGER.info(
        "event=predict_completed status=%s warning_code=%s weight_grams=%s top_k=%s",
        result.status,
        result.anomaly.warning_code if result.anomaly else None,
        result.weight_grams,
        top_k,
    )
    response = PredictionResponse(
        status=result.status,
        message=result.message,
        weight_grams=result.weight_grams,
        total_price=result.total_price,
        product=_to_match_response(result.product),
        top_matches=[_to_match_response(match) for match in result.top_matches],
        anomaly_detected=bool(result.anomaly and result.anomaly.blocked),
        warning_code=result.anomaly.warning_code if result.anomaly else None,
        crop=_to_crop_response(result.crop),
        embedding_dim=result.embedding_dim,
        pipeline_steps=result.pipeline_steps,
    )
    history = getattr(request.app.state, "prediction_history", None)
    if history is not None:
        record = history.record(
            response,
            payload,
            filename=image.filename,
            content_type=image.content_type,
            top_k=top_k,
        )
        response.prediction_id = str(record["prediction_id"])
        response.created_at = str(record["created_at"])
    return response


@router.get("/health", response_model=HealthResponse)
async def healthcheck(pipeline: RecognitionPipeline = Depends(get_pipeline)) -> HealthResponse:
    return HealthResponse(**pipeline.health_status())


def _to_match_response(match) -> MatchResponse | None:
    if match is None:
        return None
    return MatchResponse(
        product_id=match.product_id,
        name=match.name,
        product_type=match.product_type,
        product_sort=match.product_sort,
        score=match.score,
        price_rub_per_kg=match.price_rub_per_kg,
        metadata=match.metadata,
    )


def _to_crop_response(crop) -> CropResponse | None:
    if crop is None:
        return None
    return CropResponse(
        bbox=crop.bbox,
        confidence=crop.confidence,
        detector_name=crop.detector_name,
        mask_applied=crop.mask_applied,
    )
