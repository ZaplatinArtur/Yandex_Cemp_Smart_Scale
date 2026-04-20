from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    image_catalog_dir: Path
    products_csv: Path
    model_checkpoint: Path
    onnx_model: Path
    vector_db_path: Path
    file_vector_store_path: Path
    vector_backend: str
    pgvector_dsn: str | None
    pgvector_table: str
    detection_model_path: Path
    hand_landmarker_path: Path
    api_title: str
    api_host: str
    api_port: int
    default_top_k: int
    price_precision: int
    build_index_on_startup: bool
    hand_detection_enabled: bool
    embedding_dim: int
    weight_stability_tolerance: float
    weight_stability_window: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            project_root=PROJECT_ROOT,
            image_catalog_dir=_as_path(os.getenv("SMART_SCALE_IMAGE_DIR"), PROJECT_ROOT / "images"),
            products_csv=_as_path(os.getenv("SMART_SCALE_PRODUCTS_CSV"), PROJECT_ROOT / "images" / "product.csv"),
            model_checkpoint=_as_path(
                os.getenv("SMART_SCALE_MODEL_PATH"),
                PROJECT_ROOT / "assets" / "models" / "fruit_embedder_final.pth",
            ),
            onnx_model=_as_path(
                os.getenv("SMART_SCALE_ONNX_PATH"),
                PROJECT_ROOT / "assets" / "models" / "fruit_embedder_final.onnx",
            ),
            vector_db_path=_as_path(
                os.getenv("SMART_SCALE_VECTOR_DB_PATH"),
                PROJECT_ROOT / "data" / "vector_db" / "fruits.db",
            ),
            file_vector_store_path=_as_path(
                os.getenv("SMART_SCALE_FILE_VECTOR_STORE_PATH"),
                PROJECT_ROOT / "data" / "vector_db" / "catalog.pkl",
            ),
            vector_backend=os.getenv("SMART_SCALE_VECTOR_BACKEND", "file").strip().lower(),
            pgvector_dsn=os.getenv("SMART_SCALE_PGVECTOR_DSN"),
            pgvector_table=os.getenv("SMART_SCALE_PGVECTOR_TABLE", "product_embeddings"),
            detection_model_path=_as_path(
                os.getenv("SMART_SCALE_DETECTION_MODEL"),
                PROJECT_ROOT / "assets" / "models" / "yolo.onnx",
            ),
            hand_landmarker_path=_as_path(
                os.getenv("SMART_SCALE_HAND_LANDMARKER_PATH"),
                PROJECT_ROOT / "assets" / "models" / "hand_landmarker.task",
            ),
            api_title=os.getenv("SMART_SCALE_API_TITLE", "Smart Scale API"),
            api_host=os.getenv("SMART_SCALE_API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("SMART_SCALE_API_PORT", "8000")),
            default_top_k=int(os.getenv("SMART_SCALE_TOP_K", "3")),
            price_precision=int(os.getenv("SMART_SCALE_PRICE_PRECISION", "2")),
            build_index_on_startup=_as_bool(os.getenv("SMART_SCALE_BUILD_INDEX"), False),
            hand_detection_enabled=_as_bool(os.getenv("SMART_SCALE_HAND_DETECTION"), True),
            embedding_dim=int(os.getenv("SMART_SCALE_EMBEDDING_DIM", "256")),
            weight_stability_tolerance=float(os.getenv("SMART_SCALE_WEIGHT_TOLERANCE", "2.0")),
            weight_stability_window=int(os.getenv("SMART_SCALE_WEIGHT_WINDOW", "5")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
