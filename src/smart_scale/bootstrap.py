from __future__ import annotations

import sys
from pathlib import Path


if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from smart_scale.config import Settings, get_settings
from smart_scale.ml.catalog_seed import PackEatCatalogSeeder
from smart_scale.ml.detection import ProductLocalizer
from smart_scale.ml.embedding import DinoV2Embedder
from smart_scale.ml.vector_store import PgVectorStore


def bootstrap_pgvector_catalog(settings: Settings | None = None) -> int:
    resolved_settings = settings or get_settings()
    _validate_bootstrap_settings(resolved_settings)

    store = PgVectorStore(
        resolved_settings.pgvector_dsn,
        dim=resolved_settings.embedding_dim,
        table=resolved_settings.pgvector_table,
        recreate_on_dimension_mismatch=True,
    )

    embedder = DinoV2Embedder(
        model_name=resolved_settings.embedding_model_name,
        checkpoint_path=resolved_settings.embedding_checkpoint_path,
        embedding_dim=resolved_settings.embedding_dim,
    )
    embedder.warmup()

    localizer = None
    if resolved_settings.catalog_yolo_enabled:
        localizer = ProductLocalizer(model_path=resolved_settings.detection_model_path)
        if not localizer.is_ready:
            raise RuntimeError(
                "YOLO localizer is not ready for catalog bootstrap."
                + (f" Reason: {localizer.failure_reason}" if localizer.failure_reason else "")
            )

    seeder = PackEatCatalogSeeder(embedder, store, localizer=localizer)
    return seeder.build(
        dataset_dir=resolved_settings.dataset_dir,
        price_source=resolved_settings.price_catalog_path,
        samples_per_sort=resolved_settings.samples_per_sort,
    )


def _validate_bootstrap_settings(settings: Settings) -> None:
    if not settings.pgvector_dsn:
        raise RuntimeError("SMART_SCALE_PGVECTOR_DSN is required for pgvector bootstrap.")
    if not settings.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {settings.dataset_dir}")
    if not settings.price_catalog_path.exists():
        raise FileNotFoundError(f"Price catalog not found: {settings.price_catalog_path}")
    if settings.samples_per_sort <= 0:
        raise ValueError("SMART_SCALE_SAMPLES_PER_SORT must be greater than zero.")


def main() -> None:
    settings = get_settings()
    indexed_items = bootstrap_pgvector_catalog(settings)
    print(
        f"Bootstrapped {indexed_items} embeddings into {settings.pgvector_table} "
        f"using {settings.samples_per_sort} samples per sort "
        f"and catalog_yolo={settings.catalog_yolo_enabled}."
    )


if __name__ == "__main__":
    main()
