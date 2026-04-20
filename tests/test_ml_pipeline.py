from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._bootstrap import PROJECT_ROOT

try:
    import numpy as np
    from PIL import Image

    from smart_scale.config import get_settings
    from smart_scale.domain import AnomalyCheckResult, CropResult, ProductMatch
    from smart_scale.ml.anomaly import HandAnomalyDetector
    from smart_scale.ml.catalog_seed import PackEatCatalogSeeder
    from smart_scale.ml.detection import ProductLocalizer
    from smart_scale.ml.embedding import DinoV2Embedder
    from smart_scale.ml.pipeline import RecognitionPipeline
    from smart_scale.ml.vector_store import FileVectorStore, PgVectorStore
except ImportError:  # pragma: no cover - environment-dependent
    np = None
    Image = None
    get_settings = None
    AnomalyCheckResult = None
    CropResult = None
    ProductMatch = None
    HandAnomalyDetector = None
    PackEatCatalogSeeder = None
    ProductLocalizer = None
    DinoV2Embedder = None
    RecognitionPipeline = None
    FileVectorStore = None
    PgVectorStore = None


class EmptyResultModel:
    def predict(self, *_args, **_kwargs):
        class Boxes:
            def __len__(self) -> int:
                return 0

        class Result:
            boxes = Boxes()
            masks = None

        return [Result()]


class FakeHandDetector:
    def __init__(self, blocked: bool) -> None:
        self.blocked = blocked

    def detect(self, _image):
        return AnomalyCheckResult(
            blocked=self.blocked,
            message="Руки найдены." if self.blocked else "Рук нет.",
            hands_count=1 if self.blocked else 0,
            warning_code="hands_detected" if self.blocked else None,
        )


class FakeLocalizer:
    def localize(self, image):
        return CropResult(
            image=image,
            bbox=(0, 0, image.size[0], image.size[1]),
            confidence=1.0,
            detector_name="fake_localizer",
            mask_applied=False,
        )


class FakeEmbedder:
    def __init__(self, vector: np.ndarray) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)
        self.is_ready = True

    def embed(self, _image) -> np.ndarray:
        return self.vector.copy()

    def embed_batch(self, images, batch_size: int = 16) -> np.ndarray:
        _ = batch_size
        return np.repeat(self.vector.reshape(1, -1), len(images), axis=0)


class FailingBatchEmbedder(FakeEmbedder):
    def embed_batch(self, images, batch_size: int = 16) -> np.ndarray:
        _ = images
        _ = batch_size
        raise RuntimeError("embedding failed")


@unittest.skipIf(np is None or Image is None, "ML runtime test dependencies are not installed.")
class MLRuntimeTests(unittest.TestCase):
    def test_hand_detector_skips_when_asset_is_missing(self) -> None:
        detector = HandAnomalyDetector(enabled=True, model_asset_path=PROJECT_ROOT / "assets" / "models" / "missing.task")
        image = Image.open(PROJECT_ROOT / "images" / "r0_10.jpg").convert("RGB")

        result = detector.detect(image)

        self.assertFalse(result.blocked)
        self.assertEqual(result.warning_code, "hand_check_skipped")

    def test_localizer_falls_back_to_full_frame_when_detection_has_no_masks(self) -> None:
        image = Image.open(PROJECT_ROOT / "images" / "r0_10.jpg").convert("RGB")
        localizer = ProductLocalizer(model=EmptyResultModel())

        result = localizer.localize(image)

        self.assertEqual(result.bbox, (0, 0, image.size[0], image.size[1]))
        self.assertFalse(result.mask_applied)
        self.assertEqual(result.image.size, image.size)

    def test_normalize_keeps_expected_dimension_and_unit_norm(self) -> None:
        normalized = DinoV2Embedder._normalize(np.array([[3.0, 4.0, 0.0]], dtype=np.float32))

        self.assertEqual(normalized.shape, (1, 3))
        self.assertAlmostEqual(float(np.linalg.norm(normalized[0])), 1.0, places=6)

    def test_file_vector_store_returns_top_matches_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot = Path(tmp_dir) / "catalog.pkl"
            store = FileVectorStore(dim=3, snapshot_path=snapshot)
            store.add_batch(
                ids=["apple_fuji", "banana_yellow", "pear_conference"],
                embeddings=np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.8, 0.2, 0.0],
                    ],
                    dtype=np.float32,
                ),
                metadata_list=[
                    {"product_type": "apple", "product_sort": "fuji", "price_rub_per_kg": 175.0},
                    {"product_type": "banana", "product_sort": "yellow", "price_rub_per_kg": 149.0},
                    {"product_type": "pear", "product_sort": "conference", "price_rub_per_kg": 251.0},
                ],
            )
            store.save()

            restored = FileVectorStore(dim=3, snapshot_path=snapshot)
            restored.load()
            matches = restored.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=2)

        self.assertEqual(restored.count(), 3)
        self.assertEqual([match.product_id for match in matches], ["apple_fuji", "pear_conference"])
        self.assertEqual(matches[0].product_type, "apple")
        self.assertEqual(matches[0].product_sort, "fuji")

    def test_pipeline_blocks_when_hands_are_detected(self) -> None:
        settings = get_settings()
        pipeline = RecognitionPipeline(
            settings=settings,
            hand_detector=FakeHandDetector(blocked=True),
            localizer=FakeLocalizer(),
            embedder=FakeEmbedder(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            vector_store=FileVectorStore(dim=3, snapshot_path=Path(tempfile.gettempdir()) / "unused.pkl"),
        )
        pipeline._search_index_ready = True
        image = Image.open(PROJECT_ROOT / "images" / "r0_10.jpg").convert("RGB")

        result = pipeline.run(image, weight_grams=120.0)

        self.assertEqual(result.status, "warning")
        self.assertEqual(result.anomaly.warning_code, "hands_detected")

    def test_pipeline_returns_error_for_non_positive_weight(self) -> None:
        settings = get_settings()
        pipeline = RecognitionPipeline(
            settings=settings,
            hand_detector=FakeHandDetector(blocked=False),
            localizer=FakeLocalizer(),
            embedder=FakeEmbedder(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            vector_store=FileVectorStore(dim=3, snapshot_path=Path(tempfile.gettempdir()) / "unused.pkl"),
        )

        result = pipeline.run(Image.open(PROJECT_ROOT / "images" / "r0_10.jpg").convert("RGB"), weight_grams=0.0)

        self.assertEqual(result.status, "error")
        self.assertIn("weight_validation_failed", result.pipeline_steps)

    def test_pipeline_returns_error_when_catalog_is_empty(self) -> None:
        settings = get_settings()
        store = FileVectorStore(dim=3, snapshot_path=Path(tempfile.gettempdir()) / "empty.pkl")
        pipeline = RecognitionPipeline(
            settings=settings,
            hand_detector=FakeHandDetector(blocked=False),
            localizer=FakeLocalizer(),
            embedder=FakeEmbedder(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            vector_store=store,
        )
        pipeline._search_index_ready = True

        result = pipeline.run(Image.open(PROJECT_ROOT / "images" / "r0_10.jpg").convert("RGB"), weight_grams=120.0)

        self.assertEqual(result.status, "error")
        self.assertIsNone(result.product)

    def test_pipeline_returns_best_match_and_top_matches(self) -> None:
        settings = get_settings()
        store = FileVectorStore(dim=3, snapshot_path=Path(tempfile.gettempdir()) / "catalog.pkl")
        store.add_batch(
            ids=["apple_fuji", "banana_yellow"],
            embeddings=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            metadata_list=[
                {"product_type": "apple", "product_sort": "fuji", "price_rub_per_kg": 150.0},
                {"product_type": "banana", "product_sort": "yellow", "price_rub_per_kg": 120.0},
            ],
        )
        pipeline = RecognitionPipeline(
            settings=settings,
            hand_detector=FakeHandDetector(blocked=False),
            localizer=FakeLocalizer(),
            embedder=FakeEmbedder(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            vector_store=store,
        )
        pipeline._search_index_ready = True

        result = pipeline.run(Image.open(PROJECT_ROOT / "images" / "r0_10.jpg").convert("RGB"), weight_grams=100.0, top_k=2)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.product.product_id, "apple_fuji")
        self.assertEqual(result.product.product_type, "apple")
        self.assertEqual(result.product.product_sort, "fuji")
        self.assertEqual([match.product_id for match in result.top_matches], ["apple_fuji", "banana_yellow"])
        self.assertAlmostEqual(result.total_price, 15.0)

    def test_packeat_catalog_seeder_uses_five_samples_per_sort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "packeat"
            price_path = Path(tmp_dir) / "prices.py"
            price_path.write_text(
                "prices = {'apple_fuji': 175.0, 'banana_yellow': 149.0}",
                encoding="utf-8",
            )

            for label in ("apple_fuji", "banana_yellow"):
                for split in ("train", "test"):
                    target_dir = dataset_root / "classification" / split / label
                    target_dir.mkdir(parents=True, exist_ok=True)
                    for index in range(3):
                        image = Image.new("RGB", (8, 8), color=(255, 0, 0))
                        image.save(target_dir / f"{label}_{split}_{index}.jpg")

            store = FileVectorStore(dim=3, snapshot_path=Path(tmp_dir) / "catalog.pkl")
            seeder = PackEatCatalogSeeder(FakeEmbedder(np.array([1.0, 0.0, 0.0], dtype=np.float32)), store)

            indexed_items = seeder.build(dataset_root, price_path, samples_per_sort=5, batch_size=2)

        self.assertEqual(indexed_items, 10)
        self.assertEqual(store.count(), 10)
        self.assertEqual(sum(1 for meta in store.metadata if meta["product_type"] == "apple"), 5)
        self.assertEqual(sum(1 for meta in store.metadata if meta["product_type"] == "banana"), 5)

    def test_packeat_catalog_seeder_keeps_existing_catalog_when_embedding_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "packeat"
            price_path = Path(tmp_dir) / "prices.py"
            price_path.write_text("prices = {'apple_fuji': 175.0}", encoding="utf-8")
            target_dir = dataset_root / "classification" / "test" / "apple_fuji"
            target_dir.mkdir(parents=True, exist_ok=True)
            for index in range(5):
                image = Image.new("RGB", (8, 8), color=(255, 0, 0))
                image.save(target_dir / f"apple_fuji_test_{index}.jpg")

            store = FileVectorStore(dim=3, snapshot_path=Path(tmp_dir) / "catalog.pkl")
            store.add_batch(
                ids=["existing_item"],
                embeddings=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                metadata_list=[{"product_type": "banana", "product_sort": "yellow", "price_rub_per_kg": 149.0}],
            )
            seeder = PackEatCatalogSeeder(FailingBatchEmbedder(np.array([1.0, 0.0, 0.0], dtype=np.float32)), store)

            with self.assertRaises(RuntimeError):
                seeder.build(dataset_root, price_path, samples_per_sort=5, batch_size=2)

        self.assertEqual(store.count(), 1)
        self.assertEqual(store.ids, ["existing_item"])

    def test_from_settings_does_not_touch_pgvector_before_warmup(self) -> None:
        settings = replace(
            get_settings(),
            vector_backend="pgvector",
            pgvector_dsn="postgresql://smart_scale:smart_scale@localhost:5433/smart_scale",
            build_index_on_startup=True,
        )

        class FakePipelineEmbedder:
            def __init__(self, *args, **kwargs) -> None:
                _ = args
                _ = kwargs

        class FakePipelineHandDetector:
            def __init__(self, *args, **kwargs) -> None:
                _ = args
                _ = kwargs
                self.is_ready = True

            def close(self) -> None:
                return None

        class FakePipelineLocalizer:
            def __init__(self, *args, **kwargs) -> None:
                _ = args
                _ = kwargs
                self.is_ready = True
                self.detector_name = "fake_localizer"
                self.failure_reason = None

        class FakePgVectorStore:
            def __init__(self, *args, **kwargs) -> None:
                _ = args
                _ = kwargs
                self.count_calls = 0

            def count(self) -> int:
                self.count_calls += 1
                return 0

        with (
            patch("smart_scale.ml.pipeline.DinoV2Embedder", FakePipelineEmbedder),
            patch("smart_scale.ml.pipeline.HandAnomalyDetector", FakePipelineHandDetector),
            patch("smart_scale.ml.pipeline.ProductLocalizer", FakePipelineLocalizer),
            patch("smart_scale.ml.pipeline.PgVectorStore", FakePgVectorStore),
            patch.object(RecognitionPipeline, "_bootstrap_pgvector_index", autospec=True) as bootstrap_mock,
        ):
            pipeline = RecognitionPipeline.from_settings(settings)

        self.assertFalse(bootstrap_mock.called)
        self.assertFalse(pipeline._search_index_ready)
        self.assertEqual(pipeline.vector_store.count_calls, 0)

    def test_pgvector_store_builds_migration_statements_for_legacy_schema(self) -> None:
        with patch("smart_scale.ml.vector_store.psycopg", object()):
            store = PgVectorStore("postgresql://example", dim=3, table="product_embeddings")

        statements = store._migration_statements({"product_id", "embedding", "metadata", "price_per_gram"})
        combined = "\n".join(statements)

        self.assertIn("ADD COLUMN product_type TEXT", combined)
        self.assertIn("ADD COLUMN product_sort TEXT", combined)
        self.assertIn("ADD COLUMN price_rub_per_kg DOUBLE PRECISION", combined)
        self.assertIn("price_per_gram * 1000.0", combined)


if __name__ == "__main__":
    unittest.main()
