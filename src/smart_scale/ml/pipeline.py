from __future__ import annotations

from pathlib import Path
from typing import Any

from smart_scale.config import Settings
from smart_scale.domain import ProductMatch, RecognitionResult
from smart_scale.ml.anomaly import HandAnomalyDetector
from smart_scale.ml.catalog_seed import PackEatCatalogSeeder
from smart_scale.ml.detection import ProductLocalizer
from smart_scale.ml.embedding import DinoV2Embedder
from smart_scale.ml.vector_store import CatalogIndexBuilder, FaissVectorStore, FileVectorStore, PgVectorStore


class RecognitionPipeline:
    def __init__(
        self,
        settings: Settings,
        hand_detector: HandAnomalyDetector,
        localizer: ProductLocalizer,
        embedder: DinoV2Embedder,
        vector_store: Any,
    ) -> None:
        self.settings = settings
        self.hand_detector = hand_detector
        self.localizer = localizer
        self.embedder = embedder
        self.vector_store = vector_store
        self._search_index_ready = False
        self._warmup_completed = False

    @classmethod
    def from_settings(cls, settings: Settings) -> "RecognitionPipeline":
        embedder = DinoV2Embedder(
            checkpoint_path=settings.model_checkpoint,
            onnx_model_path=settings.onnx_model,
            model_name=settings.embedding_model_name,
            embedding_dim=settings.embedding_dim,
        )
        hand_detector = HandAnomalyDetector(
            enabled=settings.hand_detection_enabled,
            model_asset_path=settings.hand_landmarker_path,
        )
        localizer = ProductLocalizer(model_path=settings.detection_model_path)

        if settings.vector_backend == "pgvector":
            if not settings.pgvector_dsn:
                raise RuntimeError("Для backend=pgvector требуется SMART_SCALE_PGVECTOR_DSN.")
            vector_store = PgVectorStore(
                settings.pgvector_dsn,
                dim=settings.embedding_dim,
                table=settings.pgvector_table,
            )
            pipeline = cls(settings, hand_detector, localizer, embedder, vector_store)
            return pipeline

        if settings.vector_backend == "faiss":
            vector_store = FaissVectorStore(dim=settings.embedding_dim, snapshot_path=settings.vector_db_path)
        else:
            vector_store = FileVectorStore(dim=settings.embedding_dim, snapshot_path=settings.file_vector_store_path)
        pipeline = cls(settings, hand_detector, localizer, embedder, vector_store)

        if pipeline._local_snapshot_exists():
            vector_store.load()
            pipeline._search_index_ready = vector_store.count() > 0
        elif settings.build_index_on_startup:
            pipeline._bootstrap_local_index()

        return pipeline

    @property
    def warmup_completed(self) -> bool:
        return self._warmup_completed

    def warmup(self) -> None:
        self._validate_runtime_components()
        self.embedder.warmup()
        self._ensure_search_index()
        self._warmup_completed = True

    def run(self, image: Any, weight_grams: float, top_k: int | None = None) -> RecognitionResult:
        if weight_grams <= 0:
            return RecognitionResult(
                status="error",
                message="Вес должен быть больше нуля.",
                weight_grams=weight_grams,
                pipeline_steps=["weight_validation_failed"],
            )

        self._ensure_search_index()
        top_k = top_k or self.settings.default_top_k
        steps = ["weight_received", "anomaly_check_started"]

        anomaly = self.hand_detector.detect(image)
        steps.append("anomaly_check_completed")
        if anomaly.blocked:
            return RecognitionResult(
                status="warning",
                message=anomaly.message,
                weight_grams=weight_grams,
                pipeline_steps=steps,
                anomaly=anomaly,
            )

        crop = self.localizer.localize(image)
        steps.append("localization_completed")

        embedding = self.embedder.embed(crop.image)
        steps.append("embedding_completed")

        matches: list[ProductMatch] = self.vector_store.search(embedding, top_k=top_k)
        steps.append("knn_search_completed")

        if not matches:
            return RecognitionResult(
                status="error",
                message="Каталог пуст или совпадения не найдены.",
                weight_grams=weight_grams,
                pipeline_steps=steps,
                anomaly=anomaly,
                crop=crop,
                embedding_dim=int(embedding.shape[0]),
            )

        best_match = matches[0]
        total_price = None
        if best_match.price_rub_per_kg is not None:
            total_price = round((weight_grams / 1000.0) * best_match.price_rub_per_kg, self.settings.price_precision)

        steps.append("response_ready")
        return RecognitionResult(
            status="ok",
            message="Товар распознан.",
            weight_grams=weight_grams,
            product=best_match,
            top_matches=matches,
            anomaly=anomaly,
            crop=crop,
            total_price=total_price,
            embedding_dim=int(embedding.shape[0]),
            pipeline_steps=steps,
        )

    def health_status(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "vector_backend": self.settings.vector_backend,
            "vector_index_ready": self._search_index_ready,
            "model_ready": self.embedder.is_ready,
            "catalog_items": self.vector_store.count() if hasattr(self.vector_store, "count") else None,
            "warmup_completed": self._warmup_completed,
            "hand_detection_enabled": self.settings.hand_detection_enabled,
            "hand_detection_ready": self.hand_detector.is_ready,
            "detection_model_ready": self.localizer.is_ready,
            "detector_name": self.localizer.detector_name,
            "embedding_backend": self.embedder.backend_name,
        }

    def _ensure_search_index(self) -> None:
        if self._search_index_ready:
            return

        if self.settings.vector_backend == "pgvector":
            current_count = self.vector_store.count()
            if current_count > 0:
                self._search_index_ready = True
                return
            if self.settings.build_index_on_startup:
                self._bootstrap_pgvector_index()
            return

        if self._local_snapshot_exists():
            self.vector_store.load()
            self._search_index_ready = self.vector_store.count() > 0
            return

        self._bootstrap_local_index()

    def _bootstrap_local_index(self) -> None:
        builder = CatalogIndexBuilder(self.embedder, self.vector_store)
        indexed_items = builder.build(
            images_dir=self.settings.image_catalog_dir,
            products_csv=self.settings.products_csv,
        )
        self.vector_store.save()
        self._search_index_ready = indexed_items > 0

    def _bootstrap_pgvector_index(self) -> None:
        builder = PackEatCatalogSeeder(self.embedder, self.vector_store)
        indexed_items = builder.build(
            dataset_dir=self.settings.dataset_dir,
            price_source=self.settings.price_catalog_path,
            samples_per_sort=self.settings.samples_per_sort,
        )
        self.vector_store.save()
        self._search_index_ready = indexed_items > 0

    def _local_snapshot_exists(self) -> bool:
        if self.settings.vector_backend == "faiss":
            return self.settings.vector_db_path.exists() and Path(str(self.settings.vector_db_path) + ".index").exists()
        return self.settings.file_vector_store_path.exists()

    def _validate_runtime_components(self) -> None:
        if not self.localizer.is_ready:
            raise RuntimeError(
                "YOLO-локализатор не готов."
                + (f" Причина: {self.localizer.failure_reason}" if self.localizer.failure_reason else "")
            )

        if self.settings.vector_backend == "pgvector" and not self.settings.pgvector_dsn:
            raise RuntimeError("Для backend=pgvector требуется SMART_SCALE_PGVECTOR_DSN.")

        if self.settings.vector_backend == "pgvector":
            if self.vector_store.count() == 0 and self.settings.build_index_on_startup:
                if not self.settings.dataset_dir.exists():
                    raise RuntimeError(
                        "Не найден датасет PackEat для первичного заполнения pgvector."
                        f" Ожидался каталог {self.settings.dataset_dir}."
                    )
                if not self.settings.price_catalog_path.exists():
                    raise RuntimeError(
                        "Не найден прайс-лист для первичного заполнения pgvector."
                        f" Ожидался файл {self.settings.price_catalog_path}."
                    )
            return

        if self.settings.vector_backend != "pgvector":
            has_snapshot = self._local_snapshot_exists()
            can_bootstrap = self.settings.image_catalog_dir.exists()
            if not has_snapshot and not can_bootstrap:
                raise RuntimeError(
                    "Локальный индекс отсутствует, а каталог изображений для bootstrap недоступен."
                    f" Ожидались {self.settings.file_vector_store_path if self.settings.vector_backend != 'faiss' else self.settings.vector_db_path}"
                    f" или каталог {self.settings.image_catalog_dir}."
                )

    def close(self) -> None:
        self.hand_detector.close()
