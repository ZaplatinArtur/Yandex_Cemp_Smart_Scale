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

    from smart_scale.config import Settings, get_settings
    from smart_scale.api.routes.ui import _resolve_train_dataset_dir
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
    Settings = None
    get_settings = None
    _resolve_train_dataset_dir = None
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


class ArrayLike:
    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class BoxOnlyResultModel:
    def predict(self, *_args, **_kwargs):
        class Boxes:
            conf = ArrayLike([0.42])
            xyxy = ArrayLike([[2.0, 3.0, 6.0, 7.0]])

            def __len__(self) -> int:
                return 1

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


class CountingEmbedder(FakeEmbedder):
    def __init__(self, vector: np.ndarray) -> None:
        super().__init__(vector)
        self.embed_calls = 0
        self.embed_image_sizes: list[tuple[int, int]] = []

    def embed(self, image) -> np.ndarray:
        self.embed_calls += 1
        self.embed_image_sizes.append(image.size)
        return super().embed(image)


class RecordingEmbedder(FakeEmbedder):
    def __init__(self, vector: np.ndarray) -> None:
        super().__init__(vector)
        self.batch_image_sizes: list[tuple[int, int]] = []
        self.batch_image_types: list[type] = []

    def embed_batch(self, images, batch_size: int = 16) -> np.ndarray:
        for image in images:
            self.batch_image_types.append(type(image))
            self.batch_image_sizes.append(image.size)
        return super().embed_batch(images, batch_size=batch_size)


class CroppingLocalizer:
    def __init__(self) -> None:
        self.calls = 0

    def localize(self, image):
        self.calls += 1
        return CropResult(
            image=image.crop((0, 0, 4, 4)),
            bbox=(0, 0, 4, 4),
            confidence=1.0,
            detector_name="cropping_localizer",
            mask_applied=True,
        )


class FailingBatchEmbedder(FakeEmbedder):
    def embed_batch(self, images, batch_size: int = 16) -> np.ndarray:
        _ = images
        _ = batch_size
        raise RuntimeError("embedding failed")


@unittest.skipIf(np is None or Image is None, "ML runtime test dependencies are not installed.")
class MLRuntimeTests(unittest.TestCase):
    def test_default_embedding_model_uses_custom_best_weights(self) -> None:
        with patch.dict("os.environ", {"SMART_SCALE_EMBEDDING_CHECKPOINT": ""}, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.embedding_checkpoint_path, None)
        self.assertEqual(settings.embedding_dim, 384)
        self.assertTrue(settings.product_localization_enabled)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "best.pt"
            checkpoint_path.write_bytes(b"placeholder")
            with patch.dict("os.environ", {"SMART_SCALE_EMBEDDING_CHECKPOINT": str(checkpoint_path)}, clear=True):
                settings = Settings.from_env()

        self.assertEqual(settings.embedding_checkpoint_path, checkpoint_path)
        self.assertEqual(settings.embedding_dim, 256)
        self.assertFalse(settings.product_localization_enabled)

        with patch.dict("os.environ", {}, clear=True):
            settings = Settings.from_env()

        default_checkpoint_path = PROJECT_ROOT / "assets" / "models" / "best.pt"
        if default_checkpoint_path.exists():
            self.assertEqual(settings.embedding_checkpoint_path, default_checkpoint_path)
            self.assertEqual(settings.embedding_dim, 256)
            self.assertFalse(settings.product_localization_enabled)
        else:
            self.assertEqual(settings.embedding_checkpoint_path, None)
            self.assertEqual(settings.embedding_dim, 384)
            self.assertTrue(settings.product_localization_enabled)

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

    def test_localizer_crops_bbox_when_detection_has_no_masks(self) -> None:
        image = Image.open(PROJECT_ROOT / "images" / "r0_10.jpg").convert("RGB")
        localizer = ProductLocalizer(model=BoxOnlyResultModel())

        result = localizer.localize(image)

        self.assertEqual(result.bbox, (2, 3, 6, 7))
        self.assertFalse(result.mask_applied)
        self.assertEqual(result.image.size, (4, 4))
        self.assertAlmostEqual(result.confidence, 0.42, places=6)

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

    def test_file_vector_store_returns_unique_sorts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FileVectorStore(dim=3, snapshot_path=Path(tmp_dir) / "catalog.pkl")
            store.add_batch(
                ids=["apple_fuji:00", "apple_fuji:01", "pear_conference:00", "banana_yellow:00"],
                embeddings=np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.99, 0.01, 0.0],
                        [0.8, 0.2, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                metadata_list=[
                    {"product_type": "apple", "product_sort": "fuji", "price_rub_per_kg": 175.0},
                    {"product_type": "apple", "product_sort": "fuji", "price_rub_per_kg": 175.0},
                    {"product_type": "pear", "product_sort": "conference", "price_rub_per_kg": 251.0},
                    {"product_type": "banana", "product_sort": "yellow", "price_rub_per_kg": 149.0},
                ],
            )

            matches = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=3)

        self.assertEqual([match.product_id for match in matches], ["apple_fuji:00", "pear_conference:00", "banana_yellow:00"])
        self.assertEqual(len({(match.product_type, match.product_sort) for match in matches}), 3)

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
            ids=["apple_fuji", "pear_conference", "banana_yellow"],
            embeddings=np.array([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            metadata_list=[
                {"product_type": "apple", "product_sort": "fuji", "price_rub_per_kg": 150.0},
                {"product_type": "pear", "product_sort": "conference", "price_rub_per_kg": 110.0},
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
        self.assertEqual([match.product_id for match in result.top_matches], ["pear_conference", "banana_yellow"])
        self.assertAlmostEqual(result.total_price, 15.0)

    def test_pipeline_skips_product_localization_when_disabled(self) -> None:
        settings = replace(get_settings(), product_localization_enabled=False)
        store = FileVectorStore(dim=3, snapshot_path=Path(tempfile.gettempdir()) / "catalog.pkl")
        store.add_batch(
            ids=["apple_fuji"],
            embeddings=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            metadata_list=[
                {"product_type": "apple", "product_sort": "fuji", "price_rub_per_kg": 150.0},
            ],
        )
        localizer = CroppingLocalizer()
        pipeline = RecognitionPipeline(
            settings=settings,
            hand_detector=FakeHandDetector(blocked=False),
            localizer=localizer,
            embedder=FakeEmbedder(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            vector_store=store,
        )
        pipeline._search_index_ready = True

        result = pipeline.run(Image.open(PROJECT_ROOT / "images" / "r0_10.jpg").convert("RGB"), weight_grams=100.0)

        self.assertEqual(result.status, "ok")
        self.assertEqual(localizer.calls, 0)
        self.assertEqual(result.crop.detector_name, "full_frame")
        self.assertEqual(result.crop.bbox, (0, 0, result.crop.image.size[0], result.crop.image.size[1]))
        self.assertIn("localization_skipped", result.pipeline_steps)

    def test_pipeline_add_catalog_example_upserts_embedding_and_saves_local_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot = Path(tmp_dir) / "catalog.pkl"
            settings = replace(get_settings(), vector_backend="file", product_localization_enabled=False)
            store = FileVectorStore(dim=3, snapshot_path=snapshot)
            embedder = CountingEmbedder(np.array([0.0, 1.0, 0.0], dtype=np.float32))
            pipeline = RecognitionPipeline(
                settings=settings,
                hand_detector=FakeHandDetector(blocked=False),
                localizer=FakeLocalizer(),
                embedder=embedder,
                vector_store=store,
            )
            image_path = Path(tmp_dir) / "apple_honey.jpg"
            image = Image.new("RGB", (8, 8), color=(255, 0, 0))
            image.save(image_path)

            match = pipeline.add_catalog_example(
                image,
                product_type="apple",
                product_sort="honey",
                price_rub_per_kg=212.5,
                product_id="apple_honey:admin",
                image_path=image_path,
            )

            restored = FileVectorStore(dim=3, snapshot_path=snapshot)
            restored.load()
            snapshot_exists = snapshot.exists()

        self.assertEqual(embedder.embed_calls, 1)
        self.assertEqual(embedder.embed_image_sizes, [(8, 8)])
        self.assertEqual(store.count(), 1)
        self.assertTrue(snapshot_exists)
        self.assertEqual(restored.count(), 1)
        self.assertEqual(match.product_id, "apple_honey:admin")
        self.assertEqual(store.metadata[0]["product_type"], "apple")
        self.assertEqual(store.metadata[0]["product_sort"], "honey")
        self.assertEqual(store.metadata[0]["price_rub_per_kg"], 212.5)

    def test_resolve_train_dataset_dir_handles_root_and_direct_train_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            train_dir = dataset_root / "train"
            train_dir.mkdir(parents=True)
            (dataset_root / "test").mkdir()
            direct_train = root / "direct" / "train"
            direct_train.mkdir(parents=True)
            plain_dataset = root / "plain"

            root_settings = replace(get_settings(), dataset_dir=dataset_root)
            direct_settings = replace(get_settings(), dataset_dir=direct_train)
            plain_settings = replace(get_settings(), dataset_dir=plain_dataset)

            resolved_root = _resolve_train_dataset_dir(root_settings)
            resolved_direct = _resolve_train_dataset_dir(direct_settings)
            resolved_plain = _resolve_train_dataset_dir(plain_settings)

        self.assertEqual(resolved_root, train_dir)
        self.assertEqual(resolved_direct, direct_train)
        self.assertEqual(resolved_plain, plain_dataset)

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

    def test_packeat_catalog_seeder_applies_localizer_before_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "packeat"
            price_path = Path(tmp_dir) / "prices.py"
            price_path.write_text("prices = {'apple_fuji': 175.0}", encoding="utf-8")

            target_dir = dataset_root / "train" / "apple_fuji"
            target_dir.mkdir(parents=True, exist_ok=True)
            for index in range(5):
                image = Image.new("RGB", (8, 8), color=(255, 0, 0))
                image.save(target_dir / f"apple_fuji_{index}.jpg")

            store = FileVectorStore(dim=3, snapshot_path=Path(tmp_dir) / "catalog.pkl")
            embedder = RecordingEmbedder(np.array([1.0, 0.0, 0.0], dtype=np.float32))
            localizer = CroppingLocalizer()
            seeder = PackEatCatalogSeeder(embedder, store, localizer=localizer)

            indexed_items = seeder.build(dataset_root, price_path, samples_per_sort=5, batch_size=2)

        self.assertEqual(indexed_items, 5)
        self.assertEqual(localizer.calls, 5)
        self.assertEqual(embedder.batch_image_sizes, [(4, 4)] * 5)
        self.assertTrue(all(image_type is Image.Image for image_type in embedder.batch_image_types))
        self.assertEqual(store.count(), 5)

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

    def test_pgvector_store_search_deduplicates_by_sort_in_sql(self) -> None:
        class FakeCursor:
            def __init__(self) -> None:
                self.query = ""
                self.params = None

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def execute(self, query, params=None) -> None:
                self.query = query
                self.params = params

            def fetchall(self):
                return [
                    ("apple_fuji:00", "apple", "fuji", 175.0, 0.99),
                    ("pear_conference:00", "pear", "conference", 251.0, 0.85),
                ]

        class FakeConnection:
            def __init__(self, cursor: FakeCursor) -> None:
                self._cursor = cursor

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def cursor(self) -> FakeCursor:
                return self._cursor

        class FakePsycopg:
            def __init__(self) -> None:
                self.cursor = FakeCursor()

            def connect(self, _dsn: str) -> FakeConnection:
                return FakeConnection(self.cursor)

        fake_psycopg = FakePsycopg()
        with patch("smart_scale.ml.vector_store.psycopg", fake_psycopg):
            store = PgVectorStore("postgresql://example", dim=3, table="product_embeddings")
            store._schema_ready = True
            matches = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=2)

        self.assertEqual([match.product_id for match in matches], ["apple_fuji:00", "pear_conference:00"])
        self.assertIn("ROW_NUMBER() OVER", fake_psycopg.cursor.query)
        self.assertIn("PARTITION BY product_type, product_sort", fake_psycopg.cursor.query)
        self.assertEqual(fake_psycopg.cursor.params[1], 2)

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
