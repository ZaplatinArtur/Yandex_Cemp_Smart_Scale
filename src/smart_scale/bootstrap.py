from __future__ import annotations

from smart_scale.config import Settings, get_settings
from smart_scale.ml.catalog_seed import PackEatCatalogSeeder
from smart_scale.ml.embedding import DinoV2Embedder
from smart_scale.ml.vector_store import PgVectorStore


def bootstrap_pgvector_catalog(settings: Settings | None = None) -> int:
    resolved_settings = settings or get_settings()
    if not resolved_settings.pgvector_dsn:
        raise RuntimeError("Для bootstrap pgvector требуется SMART_SCALE_PGVECTOR_DSN.")

    embedder = DinoV2Embedder(
        checkpoint_path=resolved_settings.model_checkpoint,
        onnx_model_path=resolved_settings.onnx_model,
        embedding_dim=resolved_settings.embedding_dim,
    )
    embedder.warmup()

    store = PgVectorStore(
        resolved_settings.pgvector_dsn,
        dim=resolved_settings.embedding_dim,
        table=resolved_settings.pgvector_table,
    )
    seeder = PackEatCatalogSeeder(embedder, store)
    return seeder.build(
        dataset_dir=resolved_settings.dataset_dir,
        price_source=resolved_settings.price_catalog_path,
        samples_per_sort=resolved_settings.samples_per_sort,
    )


def main() -> None:
    settings = get_settings()
    indexed_items = bootstrap_pgvector_catalog(settings)
    print(
        f"Bootstrapped {indexed_items} embeddings into {settings.pgvector_table} "
        f"using {settings.samples_per_sort} samples per sort."
    )


if __name__ == "__main__":
    main()
