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


def _optional_path(value: str | None, default: Path | None = None) -> Path | None:
    if value is not None:
        if not value.strip():
            return None
        return _as_path(value, PROJECT_ROOT)
    if default is None or not default.exists():
        return None
    return default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    image_catalog_dir: Path
    products_csv: Path
    dataset_dir: Path
    price_catalog_path: Path
    embedding_model_name: str
    embedding_checkpoint_path: Path | None
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
    product_localization_enabled: bool
    catalog_yolo_enabled: bool
    embedding_dim: int
    samples_per_sort: int
    weight_stability_tolerance: float
    weight_stability_window: int

    @classmethod
    def from_env(cls) -> "Settings":
        embedding_checkpoint_path = _optional_path(
            os.getenv("SMART_SCALE_EMBEDDING_CHECKPOINT"),
            PROJECT_ROOT / "assets" / "models" / "best.pt",
        )
        default_embedding_dim = "256" if embedding_checkpoint_path else "384"
        default_product_localization = embedding_checkpoint_path is None

        # Resolve detection model and prefer a quantized version if available.
        det_env = os.getenv("SMART_SCALE_DETECTION_MODEL")
        det_default = _as_path(det_env, PROJECT_ROOT / "assets" / "models" / "yolo.onnx")
        det_quant_env = os.getenv("SMART_SCALE_DETECTION_QUANT_PATH")
        if det_quant_env:
            detection_model_path = _as_path(det_quant_env, det_default)
        else:
            cand_det = det_default.with_name(det_default.stem + "_int8" + det_default.suffix)
            if cand_det.exists():
                detection_model_path = cand_det
            else:
                detection_model_path = det_default

        return cls(
            project_root=PROJECT_ROOT,
            image_catalog_dir=_as_path(os.getenv("SMART_SCALE_IMAGE_DIR"), PROJECT_ROOT / "images"),
            products_csv=_as_path(os.getenv("SMART_SCALE_PRODUCTS_CSV"), PROJECT_ROOT / "images" / "product.csv"),
            dataset_dir=_as_path(
                os.getenv("SMART_SCALE_DATASET_DIR"),
                PROJECT_ROOT / "varieties_classification_dataset",
            ),
            price_catalog_path=_as_path(
                os.getenv("SMART_SCALE_PRICE_CATALOG"),
                PROJECT_ROOT / "data" / "product_prices.py",
            ),
            embedding_model_name=os.getenv("SMART_SCALE_EMBEDDING_MODEL_NAME", "facebook/dinov2-small"),
            embedding_checkpoint_path=embedding_checkpoint_path,
            vector_db_path=_as_path(
                os.getenv("SMART_SCALE_VECTOR_DB_PATH"),
                PROJECT_ROOT / "data" / "vector_db" / "fruits.db",
            ),
            file_vector_store_path=_as_path(
                os.getenv("SMART_SCALE_FILE_VECTOR_STORE_PATH"),
                PROJECT_ROOT / "data" / "vector_db" / "catalog.pkl",
            ),
            vector_backend=os.getenv("SMART_SCALE_VECTOR_BACKEND", "pgvector").strip().lower(),
            pgvector_dsn=os.getenv(
                "SMART_SCALE_PGVECTOR_DSN",
                "postgresql://smart_scale:smart_scale@localhost:5433/smart_scale",
            ),
            pgvector_table=os.getenv("SMART_SCALE_PGVECTOR_TABLE", "product_embeddings"),
            detection_model_path=detection_model_path,
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
            product_localization_enabled=_as_bool(
                os.getenv("SMART_SCALE_PRODUCT_LOCALIZATION"),
                default_product_localization,
            ),
            catalog_yolo_enabled=_as_bool(os.getenv("SMART_SCALE_CATALOG_YOLO"), False),
            embedding_dim=int(os.getenv("SMART_SCALE_EMBEDDING_DIM", default_embedding_dim)),
            samples_per_sort=int(os.getenv("SMART_SCALE_SAMPLES_PER_SORT", "5")),
            weight_stability_tolerance=float(os.getenv("SMART_SCALE_WEIGHT_TOLERANCE", "2.0")),
            weight_stability_window=int(os.getenv("SMART_SCALE_WEIGHT_WINDOW", "5")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
