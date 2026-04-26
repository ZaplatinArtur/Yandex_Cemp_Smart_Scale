from __future__ import annotations

from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

from tests._bootstrap import PROJECT_ROOT

try:
    from PIL import Image
    from fastapi.testclient import TestClient

    from smart_scale.api.app import create_app
    from smart_scale.config import get_settings
    from smart_scale.domain import AnomalyCheckResult, CropResult, ProductMatch, RecognitionResult
except ImportError:  # pragma: no cover - environment-dependent
    Image = None
    TestClient = None
    create_app = None
    get_settings = None
    AnomalyCheckResult = None
    CropResult = None
    ProductMatch = None
    RecognitionResult = None


class FakePipeline:
    def __init__(
        self,
        *,
        result: RecognitionResult | None = None,
        health: dict | None = None,
        warmup_error: Exception | None = None,
        run_error: Exception | None = None,
        settings=None,
    ) -> None:
        self._result = result
        self.settings = settings or get_settings()
        self._health = health or {
            "status": "ok",
            "vector_backend": "pgvector",
            "vector_index_ready": True,
            "model_ready": True,
            "catalog_items": 2,
            "warmup_completed": True,
            "hand_detection_enabled": True,
            "hand_detection_ready": False,
            "product_localization_enabled": False,
            "detection_model_ready": True,
            "detector_name": "fake_detector",
            "embedding_backend": "torch",
        }
        self._warmup_error = warmup_error
        self._run_error = run_error
        self.warmup_called = False
        self.close_called = False
        self.added_examples = []

    def warmup(self) -> None:
        self.warmup_called = True
        if self._warmup_error is not None:
            raise self._warmup_error

    def run(self, _image, _weight_grams, _top_k=None):
        if self._run_error is not None:
            raise self._run_error
        return self._result

    def health_status(self) -> dict:
        return dict(self._health)

    def add_catalog_example(self, image, **kwargs):
        self.added_examples.append({"image_size": image.size, **kwargs})
        return ProductMatch(
            product_id=kwargs["product_id"],
            product_type=kwargs["product_type"],
            product_sort=kwargs["product_sort"],
            score=1.0,
            price_rub_per_kg=kwargs["price_rub_per_kg"],
            metadata={
                "product_type": kwargs["product_type"],
                "product_sort": kwargs["product_sort"],
                "price_rub_per_kg": kwargs["price_rub_per_kg"],
                "path": str(kwargs.get("image_path")),
            },
        )

    def close(self) -> None:
        self.close_called = True


@unittest.skipIf(TestClient is None or Image is None, "FastAPI test dependencies are not installed.")
class APITests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = get_settings()

    def test_root_endpoint_lists_entrypoints(self) -> None:
        pipeline = FakePipeline(result=_ok_result())
        app = create_app(settings=self.settings, pipeline_factory=lambda _settings: pipeline)

        with TestClient(app) as client:
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["predict"], "/api/predict")
        self.assertTrue(pipeline.warmup_called)
        self.assertTrue(pipeline.close_called)

    def test_health_reflects_warmup_state(self) -> None:
        pipeline = FakePipeline(result=_ok_result())
        app = create_app(settings=self.settings, pipeline_factory=lambda _settings: pipeline)

        with TestClient(app) as client:
            response = client.get("/api/health")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["warmup_completed"])
        self.assertEqual(payload["embedding_backend"], "torch")

    def test_ui_page_renders(self) -> None:
        pipeline = FakePipeline(result=_ok_result())
        app = create_app(settings=self.settings, pipeline_factory=lambda _settings: pipeline)

        with TestClient(app) as client:
            response = client.get("/ui")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Smart Scale", response.text)
        self.assertIn('id="latestResultImage"', response.text)
        self.assertIn('id="workspacePanel"', response.text)
        self.assertIn('id="confirmYesBtn"', response.text)
        self.assertIn('id="confirmNoBtn"', response.text)
        self.assertIn('id="adminAddBtn"', response.text)
        self.assertIn('id="adminAuthForm"', response.text)
        self.assertIn('id="adminAuthBtn"', response.text)
        self.assertIn('id="adminAddForm"', response.text)
        self.assertIn("/api/admin/verify", response.text)
        self.assertIn("/api/admin/catalog/examples", response.text)
        self.assertIn("/api/serve_image?p=mem.jpg", response.text)
        self.assertIn("/api/predictions/latest", response.text)
        self.assertIn("function pollLatestPrediction", response.text)
        self.assertIn("function renderIdleResult", response.text)
        self.assertIn("function verifyAdminAccess", response.text)
        self.assertNotIn('id="predictForm"', response.text)
        self.assertNotIn('id="topkInput"', response.text)
        self.assertNotIn('id="filePicker"', response.text)
        self.assertNotIn("fetch('/api/predict'", response.text)
        self.assertNotIn("Добавление в train и БД", response.text)

    def test_catalog_varieties_merges_price_catalog_and_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            price_path = root / "prices.py"
            price_path.write_text("prices = {'apple_fuji': 175.0, 'banana_yellow': 149.0}", encoding="utf-8")
            dataset_dir = root / "dataset"
            (dataset_dir / "apple_fuji").mkdir(parents=True)
            (dataset_dir / "pear_conference").mkdir()
            settings = replace(
                self.settings,
                price_catalog_path=price_path,
                dataset_dir=dataset_dir,
                feedback_dir=root / "feedback",
                prediction_history_dir=root / "predictions",
            )
            pipeline = FakePipeline(result=_ok_result(), settings=settings)
            app = create_app(settings=settings, pipeline_factory=lambda _settings: pipeline)

            with TestClient(app) as client:
                response = client.get("/api/catalog/varieties")

        payload = response.json()
        labels = {item["label"]: item for item in payload["items"]}
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["count"], 3)
        self.assertEqual(labels["apple_fuji"]["sources"], ["price", "dataset"])
        self.assertEqual(labels["banana_yellow"]["price_rub_per_kg"], 149.0)
        self.assertEqual(labels["pear_conference"]["sources"], ["dataset"])

    def test_incorrect_feedback_saves_image_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            price_path = root / "prices.py"
            price_path.write_text("prices = {'apple_fuji': 175.0}", encoding="utf-8")
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            feedback_dir = root / "feedback"
            settings = replace(
                self.settings,
                price_catalog_path=price_path,
                dataset_dir=dataset_dir,
                feedback_dir=feedback_dir,
                prediction_history_dir=root / "predictions",
            )
            pipeline = FakePipeline(result=_ok_result(), settings=settings)
            app = create_app(settings=settings, pipeline_factory=lambda _settings: pipeline)

            with TestClient(app) as client:
                response = client.post(
                    "/api/feedback/incorrect",
                    data={
                        "correct_label": "apple_fuji",
                        "selected_from": "catalog",
                        "prediction_json": json.dumps({"product": {"product_id": "banana_yellow"}}),
                    },
                    files={"image": ("sample.jpg", _image_bytes(), "image/jpeg")},
                )

            payload = response.json()
            image_path = Path(payload["image_path"])
            metadata_path = Path(payload["metadata_path"])
            image_exists = image_path.exists()
            metadata_exists = metadata_path.exists()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(response.status_code, 200)
        self.assertTrue(image_exists)
        self.assertTrue(metadata_exists)
        self.assertEqual(metadata["correct_label"], "apple_fuji")
        self.assertEqual(metadata["selected_from"], "catalog")
        self.assertEqual(metadata["prediction"]["product"]["product_id"], "banana_yellow")

    def test_predict_accepts_valid_multipart_request(self) -> None:
        pipeline = FakePipeline(result=_ok_result())
        app = create_app(settings=self.settings, pipeline_factory=lambda _settings: pipeline)

        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                data={"weight_grams": "125.0", "top_k": "2"},
                files={"image": ("fruit.jpg", _image_bytes(), "image/jpeg")},
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["product"]["product_id"], "apple_fuji")
        self.assertEqual(payload["product"]["product_type"], "apple")
        self.assertEqual(payload["product"]["product_sort"], "fuji")
        self.assertIsNotNone(payload["prediction_id"])

    def test_predict_accepts_top_k_ten(self) -> None:
        pipeline = FakePipeline(result=_ok_result())
        app = create_app(settings=self.settings, pipeline_factory=lambda _settings: pipeline)

        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                data={"weight_grams": "125.0", "top_k": "10"},
                files={"image": ("fruit.jpg", _image_bytes(), "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)

    def test_predict_rejects_top_k_above_ten(self) -> None:
        pipeline = FakePipeline(result=_ok_result())
        app = create_app(settings=self.settings, pipeline_factory=lambda _settings: pipeline)

        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                data={"weight_grams": "125.0", "top_k": "11"},
                files={"image": ("fruit.jpg", _image_bytes(), "image/jpeg")},
            )

        self.assertEqual(response.status_code, 422)

    def test_predict_updates_latest_prediction_for_external_clients(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = replace(self.settings, prediction_history_dir=root / "predictions")
            pipeline = FakePipeline(result=_ok_result(), settings=settings)
            app = create_app(settings=settings, pipeline_factory=lambda _settings: pipeline)

            with TestClient(app) as client:
                predict_response = client.post(
                    "/api/predict",
                    data={"weight_grams": "125.0", "top_k": "2"},
                    files={"image": ("fruit.jpg", _image_bytes(), "image/jpeg")},
                )
                latest_response = client.get("/api/predictions/latest")

            predict_payload = predict_response.json()
            latest_payload = latest_response.json()
            stored_image_exists = Path(latest_payload["image_path"]).exists()

        self.assertEqual(predict_response.status_code, 200)
        self.assertEqual(latest_response.status_code, 200)
        self.assertEqual(latest_payload["status"], "ok")
        self.assertEqual(latest_payload["prediction_id"], predict_payload["prediction_id"])
        self.assertEqual(latest_payload["prediction"]["product"]["product_id"], "apple_fuji")
        self.assertTrue(stored_image_exists)

    def test_predict_updates_latest_prediction_with_empty_state_for_zero_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = replace(self.settings, prediction_history_dir=root / "predictions")
            pipeline = FakePipeline(result=_empty_result(), settings=settings)
            app = create_app(settings=settings, pipeline_factory=lambda _settings: pipeline)

            with TestClient(app) as client:
                predict_response = client.post(
                    "/api/predict",
                    data={"weight_grams": "0.0", "top_k": "2"},
                    files={"image": ("fruit.jpg", _image_bytes(), "image/jpeg")},
                )
                latest_response = client.get("/api/predictions/latest")

            predict_payload = predict_response.json()
            latest_payload = latest_response.json()

        self.assertEqual(predict_response.status_code, 200)
        self.assertEqual(predict_payload["status"], "empty")
        self.assertEqual(latest_response.status_code, 200)
        self.assertEqual(latest_payload["status"], "ok")
        self.assertEqual(latest_payload["prediction_id"], predict_payload["prediction_id"])
        self.assertEqual(latest_payload["prediction"]["status"], "empty")
        self.assertEqual(latest_payload["prediction"]["weight_grams"], 0.0)
        self.assertIsNone(latest_payload["prediction"]["product"])

    def test_prediction_image_endpoint_returns_original_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = replace(self.settings, prediction_history_dir=root / "predictions")
            pipeline = FakePipeline(result=_ok_result(), settings=settings)
            app = create_app(settings=settings, pipeline_factory=lambda _settings: pipeline)

            with TestClient(app) as client:
                predict_response = client.post(
                    "/api/predict",
                    data={"weight_grams": "125.0", "top_k": "2"},
                    files={"image": ("fruit.jpg", _image_bytes(), "image/jpeg")},
                )
                prediction_id = predict_response.json()["prediction_id"]
                image_response = client.get(f"/api/predictions/{prediction_id}/image")

            original_image = Image.open(io.BytesIO(image_response.content))

        self.assertEqual(predict_response.status_code, 200)
        self.assertEqual(image_response.status_code, 200)
        self.assertEqual(image_response.headers["content-type"], "image/jpeg")
        self.assertEqual(original_image.size, (8, 8))

    def test_prediction_crop_endpoint_returns_saved_crop_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = replace(self.settings, prediction_history_dir=root / "predictions")
            pipeline = FakePipeline(
                result=_ok_result(crop_image=Image.new("RGB", (4, 5), color=(0, 0, 255))),
                settings=settings,
            )
            app = create_app(settings=settings, pipeline_factory=lambda _settings: pipeline)

            with TestClient(app) as client:
                predict_response = client.post(
                    "/api/predict",
                    data={"weight_grams": "125.0", "top_k": "2"},
                    files={"image": ("fruit.jpg", _image_bytes(), "image/jpeg")},
                )
                prediction_id = predict_response.json()["prediction_id"]
                crop_response = client.get(f"/api/predictions/{prediction_id}/crop")

            crop_image = Image.open(io.BytesIO(crop_response.content))

        self.assertEqual(predict_response.status_code, 200)
        self.assertEqual(crop_response.status_code, 200)
        self.assertEqual(crop_response.headers["content-type"], "image/jpeg")
        self.assertEqual(crop_image.size, (4, 5))

    def test_incorrect_feedback_can_use_stored_prediction_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            price_path = root / "prices.py"
            price_path.write_text("prices = {'apple_fuji': 175.0}", encoding="utf-8")
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            settings = replace(
                self.settings,
                price_catalog_path=price_path,
                dataset_dir=dataset_dir,
                feedback_dir=root / "feedback",
                prediction_history_dir=root / "predictions",
            )
            pipeline = FakePipeline(result=_ok_result(), settings=settings)
            app = create_app(settings=settings, pipeline_factory=lambda _settings: pipeline)

            with TestClient(app) as client:
                predict_response = client.post(
                    "/api/predict",
                    data={"weight_grams": "125.0"},
                    files={"image": ("fruit.jpg", _image_bytes(), "image/jpeg")},
                )
                feedback_response = client.post(
                    "/api/feedback/incorrect",
                    data={
                        "correct_label": "apple_fuji",
                        "selected_from": "catalog",
                        "prediction_id": predict_response.json()["prediction_id"],
                    },
                )

            feedback_payload = feedback_response.json()
            metadata_path = Path(feedback_payload["metadata_path"])
            metadata_exists = metadata_path.exists()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(feedback_response.status_code, 200)
        self.assertTrue(metadata_exists)
        self.assertEqual(metadata["prediction_id"], predict_response.json()["prediction_id"])
        self.assertEqual(metadata["prediction"]["product"]["product_id"], "apple_fuji")

    def test_admin_catalog_example_requires_valid_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            price_path = root / "prices.py"
            price_path.write_text("prices = {'apple_fuji': 175.0}", encoding="utf-8")
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            settings = replace(
                self.settings,
                admin_token="secret",
                price_catalog_path=price_path,
                dataset_dir=dataset_dir,
                prediction_history_dir=root / "predictions",
            )
            pipeline = FakePipeline(result=_ok_result(), settings=settings)
            app = create_app(settings=settings, pipeline_factory=lambda _settings: pipeline)

            with TestClient(app) as client:
                response = client.post(
                    "/api/admin/catalog/examples",
                    data={
                        "admin_token": "wrong",
                        "product_type": "apple",
                        "product_sort": "honey",
                        "price_rub_per_kg": "212.50",
                    },
                    files={"image": ("sample.jpg", _image_bytes(), "image/jpeg")},
                )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Неверный пароль администратора")
        self.assertEqual(pipeline.added_examples, [])

    def test_admin_verify_accepts_valid_token(self) -> None:
        settings = replace(self.settings, admin_token="secret")
        pipeline = FakePipeline(result=_ok_result(), settings=settings)
        app = create_app(settings=settings, pipeline_factory=lambda _settings: pipeline)

        with TestClient(app) as client:
            response = client.post("/api/admin/verify", data={"admin_token": "secret"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_admin_verify_rejects_invalid_token(self) -> None:
        settings = replace(self.settings, admin_token="secret")
        pipeline = FakePipeline(result=_ok_result(), settings=settings)
        app = create_app(settings=settings, pipeline_factory=lambda _settings: pipeline)

        with TestClient(app) as client:
            response = client.post("/api/admin/verify", data={"admin_token": "wrong"})

        self.assertEqual(response.status_code, 403)
        self.assertEqual(
            response.json()["detail"],
            "\u041d\u0435\u0432\u0435\u0440\u043d\u044b\u0439 \u043f\u0430\u0440\u043e\u043b\u044c \u0430\u0434\u043c\u0438\u043d\u0438\u0441\u0442\u0440\u0430\u0442\u043e\u0440\u0430",
        )

    def test_admin_verify_rejects_missing_config(self) -> None:
        settings = replace(self.settings, admin_token=None)
        pipeline = FakePipeline(result=_ok_result(), settings=settings)
        app = create_app(settings=settings, pipeline_factory=lambda _settings: pipeline)

        with TestClient(app) as client:
            response = client.post("/api/admin/verify", data={"admin_token": "secret"})

        self.assertEqual(response.status_code, 403)
        self.assertEqual(
            response.json()["detail"],
            "\u041f\u0430\u0440\u043e\u043b\u044c \u0430\u0434\u043c\u0438\u043d\u0438\u0441\u0442\u0440\u0430\u0442\u043e\u0440\u0430 \u043d\u0435 \u043d\u0430\u0441\u0442\u0440\u043e\u0435\u043d",
        )

    def test_admin_catalog_example_rejects_unconfigured_admin_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            price_path = root / "prices.py"
            price_path.write_text("prices = {'apple_fuji': 175.0}", encoding="utf-8")
            settings = replace(
                self.settings,
                admin_token=None,
                price_catalog_path=price_path,
                dataset_dir=root / "dataset",
                prediction_history_dir=root / "predictions",
            )
            pipeline = FakePipeline(result=_ok_result(), settings=settings)
            app = create_app(settings=settings, pipeline_factory=lambda _settings: pipeline)

            with TestClient(app) as client:
                response = client.post(
                    "/api/admin/catalog/examples",
                    data={
                        "admin_token": "secret",
                        "product_type": "apple",
                        "product_sort": "honey",
                        "price_rub_per_kg": "212.50",
                    },
                    files={"image": ("sample.jpg", _image_bytes(), "image/jpeg")},
                )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Пароль администратора не настроен")
        self.assertEqual(pipeline.added_examples, [])

    def test_admin_catalog_example_rejects_invalid_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            price_path = root / "prices.py"
            price_path.write_text("prices = {'apple_fuji': 175.0}", encoding="utf-8")
            settings = replace(
                self.settings,
                admin_token="secret",
                price_catalog_path=price_path,
                dataset_dir=root / "dataset",
                prediction_history_dir=root / "predictions",
            )
            pipeline = FakePipeline(result=_ok_result(), settings=settings)
            app = create_app(settings=settings, pipeline_factory=lambda _settings: pipeline)

            with TestClient(app) as client:
                slug_response = client.post(
                    "/api/admin/catalog/examples",
                    data={
                        "admin_token": "secret",
                        "product_type": "apple pie",
                        "product_sort": "honey",
                        "price_rub_per_kg": "212.50",
                    },
                )
                price_response = client.post(
                    "/api/admin/catalog/examples",
                    data={
                        "admin_token": "secret",
                        "product_type": "apple",
                        "product_sort": "honey",
                        "price_rub_per_kg": "0",
                    },
                )

        self.assertEqual(slug_response.status_code, 400)
        self.assertEqual(slug_response.json()["detail"], "Invalid product_type")
        self.assertEqual(price_response.status_code, 400)
        self.assertEqual(price_response.json()["detail"], "price_rub_per_kg must be greater than zero")
        self.assertEqual(pipeline.added_examples, [])

    def test_admin_catalog_example_adds_new_label_to_train_price_catalog_and_vector_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            price_path = root / "prices.py"
            price_path.write_text("prices = {'apple_fuji': 175.0}", encoding="utf-8")
            dataset_dir = root / "dataset"
            (dataset_dir / "train").mkdir(parents=True)
            (dataset_dir / "test").mkdir()
            settings = replace(
                self.settings,
                admin_token="secret",
                price_catalog_path=price_path,
                dataset_dir=dataset_dir,
                prediction_history_dir=root / "predictions",
            )
            pipeline = FakePipeline(result=_ok_result(), settings=settings)
            app = create_app(settings=settings, pipeline_factory=lambda _settings: pipeline)

            with TestClient(app) as client:
                response = client.post(
                    "/api/admin/catalog/examples",
                    data={
                        "admin_token": "secret",
                        "product_type": "apple",
                        "product_sort": "honey",
                        "price_rub_per_kg": "212.50",
                    },
                    files={"image": ("sample.jpg", _image_bytes(), "image/jpeg")},
                )

            payload = response.json()
            image_path = Path(payload["image_path"])
            image_exists = image_path.exists()
            image_parent = image_path.parent.resolve()
            expected_parent = (dataset_dir / "train" / "apple_honey").resolve()
            price_text = price_path.read_text(encoding="utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["label"], "apple_honey")
        self.assertTrue(payload["catalog_updated"])
        self.assertTrue(image_exists)
        self.assertEqual(image_parent, expected_parent)
        self.assertIn('"apple_honey": 212.50', price_text)
        self.assertEqual(len(pipeline.added_examples), 1)
        self.assertEqual(pipeline.added_examples[0]["product_id"], payload["product_id"])
        self.assertEqual(pipeline.added_examples[0]["price_rub_per_kg"], 212.5)

    def test_admin_catalog_example_rejects_existing_label_price_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            price_path = root / "prices.py"
            price_path.write_text("prices = {'apple_fuji': 175.0}", encoding="utf-8")
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            settings = replace(
                self.settings,
                admin_token="secret",
                price_catalog_path=price_path,
                dataset_dir=dataset_dir,
                prediction_history_dir=root / "predictions",
            )
            pipeline = FakePipeline(result=_ok_result(), settings=settings)
            app = create_app(settings=settings, pipeline_factory=lambda _settings: pipeline)

            with TestClient(app) as client:
                response = client.post(
                    "/api/admin/catalog/examples",
                    data={
                        "admin_token": "secret",
                        "product_type": "apple",
                        "product_sort": "fuji",
                        "price_rub_per_kg": "199.00",
                    },
                    files={"image": ("sample.jpg", _image_bytes(), "image/jpeg")},
                )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], "Price for this product variety already exists and differs")
        self.assertEqual(pipeline.added_examples, [])

    def test_admin_catalog_example_can_use_stored_prediction_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            price_path = root / "prices.py"
            price_path.write_text("prices = {'apple_fuji': 175.0}", encoding="utf-8")
            train_dir = root / "train"
            train_dir.mkdir()
            settings = replace(
                self.settings,
                admin_token="secret",
                price_catalog_path=price_path,
                dataset_dir=train_dir,
                prediction_history_dir=root / "predictions",
            )
            pipeline = FakePipeline(result=_ok_result(), settings=settings)
            app = create_app(settings=settings, pipeline_factory=lambda _settings: pipeline)

            with TestClient(app) as client:
                predict_response = client.post(
                    "/api/predict",
                    data={"weight_grams": "125.0"},
                    files={"image": ("fruit.jpg", _image_bytes(), "image/jpeg")},
                )
                response = client.post(
                    "/api/admin/catalog/examples",
                    data={
                        "admin_token": "secret",
                        "product_type": "pear",
                        "product_sort": "anjou",
                        "price_rub_per_kg": "180.00",
                        "prediction_id": predict_response.json()["prediction_id"],
                    },
                )

            payload = response.json()
            image_path = Path(payload["image_path"])
            image_exists = image_path.exists()
            image_parent = image_path.parent.resolve()
            expected_parent = (train_dir / "pear_anjou").resolve()

        self.assertEqual(predict_response.status_code, 200)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(image_exists)
        self.assertEqual(image_parent, expected_parent)
        self.assertEqual(pipeline.added_examples[0]["image_size"], (8, 8))

    def test_predict_rejects_empty_file(self) -> None:
        pipeline = FakePipeline(result=_ok_result())
        app = create_app(settings=self.settings, pipeline_factory=lambda _settings: pipeline)

        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                data={"weight_grams": "125.0"},
                files={"image": ("fruit.jpg", b"", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Файл изображения пустой.")

    def test_predict_rejects_invalid_image_payload(self) -> None:
        pipeline = FakePipeline(result=_ok_result())
        app = create_app(settings=self.settings, pipeline_factory=lambda _settings: pipeline)

        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                data={"weight_grams": "125.0"},
                files={"image": ("fruit.jpg", b"not-an-image", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Не удалось распознать изображение.")

    def test_predict_returns_empty_state_for_zero_weight(self) -> None:
        pipeline = FakePipeline(result=_empty_result())
        app = create_app(settings=self.settings, pipeline_factory=lambda _settings: pipeline)

        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                data={"weight_grams": "0.0"},
                files={"image": ("fruit.jpg", _image_bytes(), "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "empty")

    def test_predict_returns_warning_when_hands_are_detected(self) -> None:
        pipeline = FakePipeline(result=_warning_result())
        app = create_app(settings=self.settings, pipeline_factory=lambda _settings: pipeline)

        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                data={"weight_grams": "50.0"},
                files={"image": ("fruit.jpg", _image_bytes(), "image/jpeg")},
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "warning")
        self.assertEqual(payload["warning_code"], "hands_detected")

    def test_predict_returns_business_error_when_no_matches_found(self) -> None:
        pipeline = FakePipeline(result=_business_error_result(message="Каталог пуст или совпадения не найдены."))
        app = create_app(settings=self.settings, pipeline_factory=lambda _settings: pipeline)

        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                data={"weight_grams": "70.0"},
                files={"image": ("fruit.jpg", _image_bytes(), "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "error")

    def test_predict_returns_503_for_infrastructure_failure(self) -> None:
        pipeline = FakePipeline(result=None, run_error=RuntimeError("backend down"))
        app = create_app(settings=self.settings, pipeline_factory=lambda _settings: pipeline)

        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                data={"weight_grams": "70.0"},
                files={"image": ("fruit.jpg", _image_bytes(), "image/jpeg")},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Пайплайн инференса временно недоступен.")

    def test_startup_fails_when_warmup_raises(self) -> None:
        pipeline = FakePipeline(result=_ok_result(), warmup_error=RuntimeError("startup failed"))
        app = create_app(settings=self.settings, pipeline_factory=lambda _settings: pipeline)

        with self.assertRaises(RuntimeError):
            with TestClient(app):
                pass


def _image_bytes() -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGB", (8, 8), color=(255, 0, 0))
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _ok_result(crop_image=None) -> RecognitionResult:
    return RecognitionResult(
        status="ok",
        message="Товар распознан.",
        weight_grams=125.0,
        product=ProductMatch(
            product_id="apple_fuji",
            product_type="apple",
            product_sort="fuji",
            score=0.99,
            price_rub_per_kg=150.0,
            metadata={"product_type": "apple", "product_sort": "fuji"},
        ),
        top_matches=[
            ProductMatch(
                product_id="apple_fuji",
                product_type="apple",
                product_sort="fuji",
                score=0.99,
                price_rub_per_kg=150.0,
                metadata={"product_type": "apple", "product_sort": "fuji"},
            ),
            ProductMatch(
                product_id="pear_conference",
                product_type="pear",
                product_sort="conference",
                score=0.75,
                price_rub_per_kg=110.0,
                metadata={"product_type": "pear", "product_sort": "conference"},
            ),
        ],
        crop=CropResult(
            image=crop_image,
            bbox=(0, 0, 8, 8),
            confidence=0.95,
            detector_name="fake_detector",
            mask_applied=True,
        ),
        total_price=18.75,
        embedding_dim=256,
        pipeline_steps=["response_ready"],
    )


def _warning_result() -> RecognitionResult:
    return RecognitionResult(
        status="warning",
        message="Обнаружены руки на платформе весов. Инференс остановлен.",
        weight_grams=50.0,
        anomaly=AnomalyCheckResult(
            blocked=True,
            message="Обнаружены руки на платформе весов. Инференс остановлен.",
            hands_count=1,
            warning_code="hands_detected",
        ),
        pipeline_steps=["anomaly_check_completed"],
    )


def _empty_result() -> RecognitionResult:
    return RecognitionResult(
        status="empty",
        message="На весах ничего нет.",
        weight_grams=0.0,
        pipeline_steps=["empty_weight_detected"],
    )


def _business_error_result(message: str = "Вес должен быть больше нуля.") -> RecognitionResult:
    return RecognitionResult(
        status="error",
        message=message,
        weight_grams=0.0,
        pipeline_steps=["weight_validation_failed"],
    )


if __name__ == "__main__":
    unittest.main()
