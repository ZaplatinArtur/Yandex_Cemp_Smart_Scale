from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import PROJECT_ROOT

try:
    import numpy as np
    from PIL import Image

    from smart_scale.config import get_settings
    from smart_scale.domain import AnomalyCheckResult, CropResult, ProductMatch
    from smart_scale.ml.anomaly import HandAnomalyDetector
    from smart_scale.ml.detection import ProductLocalizer
    from smart_scale.ml.embedding import DinoV2Embedder
    from smart_scale.ml.pipeline import RecognitionPipeline
    from smart_scale.ml.vector_store import FileVectorStore
except ImportError:  # pragma: no cover - environment-dependent
    np = None
    Image = None
    get_settings = None
    AnomalyCheckResult = None
    CropResult = None
    ProductMatch = None
    HandAnomalyDetector = None
    ProductLocalizer = None
    DinoV2Embedder = None
    RecognitionPipeline = None
    FileVectorStore = None


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
                ids=["apple", "banana", "pear"],
                embeddings=np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.8, 0.2, 0.0],
                    ],
                    dtype=np.float32,
                ),
                metadata_list=[
                    {"name": "Apple", "price_per_gram": 1.0},
                    {"name": "Banana", "price_per_gram": 2.0},
                    {"name": "Pear", "price_per_gram": 3.0},
                ],
            )
            store.save()

            restored = FileVectorStore(dim=3, snapshot_path=snapshot)
            restored.load()
            matches = restored.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=2)

        self.assertEqual(restored.count(), 3)
        self.assertEqual([match.product_id for match in matches], ["apple", "pear"])

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
            ids=["apple", "banana"],
            embeddings=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            metadata_list=[
                {"name": "Apple", "price_per_gram": 1.5},
                {"name": "Banana", "price_per_gram": 2.0},
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
        self.assertEqual(result.product.product_id, "apple")
        self.assertEqual([match.product_id for match in result.top_matches], ["apple", "banana"])
        self.assertAlmostEqual(result.total_price, 150.0)


if __name__ == "__main__":
    unittest.main()
