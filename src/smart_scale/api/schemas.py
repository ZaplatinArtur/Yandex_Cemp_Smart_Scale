from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MatchResponse(BaseModel):
    product_id: str
    name: str
    product_type: str
    product_sort: str
    score: float
    price_rub_per_kg: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CropResponse(BaseModel):
    bbox: tuple[int, int, int, int]
    confidence: float
    detector_name: str
    mask_applied: bool = False


class PredictionResponse(BaseModel):
    prediction_id: str | None = None
    created_at: str | None = None
    status: Literal["ok", "warning", "error"]
    message: str
    weight_grams: float
    total_price: float | None = None
    product: MatchResponse | None = None
    top_matches: list[MatchResponse] = Field(default_factory=list)
    anomaly_detected: bool = False
    warning_code: str | None = None
    crop: CropResponse | None = None
    embedding_dim: int | None = None
    pipeline_steps: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    vector_backend: str
    vector_index_ready: bool
    model_ready: bool
    catalog_items: int | None = None
    warmup_completed: bool = False
    hand_detection_enabled: bool = False
    hand_detection_ready: bool = False
    product_localization_enabled: bool = True
    detection_model_ready: bool = False
    detector_name: str | None = None
    embedding_backend: str | None = None
