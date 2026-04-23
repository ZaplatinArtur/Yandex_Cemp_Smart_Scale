from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class AnomalyCheckResult:
    blocked: bool
    message: str
    hands_count: int = 0
    warning_code: str | None = None


@dataclass(slots=True)
class CropResult:
    image: Any
    bbox: tuple[int, int, int, int]
    confidence: float
    detector_name: str
    mask_applied: bool = False


@dataclass(slots=True)
class ProductMatch:
    product_id: str
    product_type: str
    product_sort: str
    score: float
    price_rub_per_kg: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.product_type}_{self.product_sort}" if self.product_sort else self.product_type


@dataclass(slots=True)
class RecognitionResult:
    status: Literal["ok", "warning", "error", "empty"]
    message: str
    weight_grams: float
    pipeline_steps: list[str] = field(default_factory=list)
    product: ProductMatch | None = None
    top_matches: list[ProductMatch] = field(default_factory=list)
    anomaly: AnomalyCheckResult | None = None
    crop: CropResult | None = None
    total_price: float | None = None
    embedding_dim: int | None = None


@dataclass(slots=True)
class WeightSnapshot:
    weight_grams: float
    stable: bool
    samples: list[float] = field(default_factory=list)
    spread: float = 0.0


@dataclass(slots=True)
class CaptureBundle:
    image: Any
    weight: WeightSnapshot
