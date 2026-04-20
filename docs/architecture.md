# Архитектура Smart Scale

## Целевой поток данных

1. `hardware.controller.SmartScaleController` ждёт стабилизации веса.
2. `hardware.camera.*` снимает кадр.
3. Фото и вес отправляются в `api.py` / FastAPI.
4. `ml.anomaly.HandAnomalyDetector` проверяет наличие рук.
5. `ml.detection.ProductLocalizer` находит объект и делает crop.
6. `ml.embedding.DinoV2Embedder` строит нормализованный эмбеддинг.
7. `ml.vector_store.*` выполняет K-NN поиск по каталогу.
8. `ml.pipeline.RecognitionPipeline` собирает итоговый ответ.
9. `api/routes/predict.py` возвращает строгий JSON.

## Слои проекта

Entry points:

- `src/smart_scale/cli.py` — основной console entrypoint (`smart-scale-api`).
- `src/smart_scale/__main__.py` — запуск через `python -m smart_scale`.
- `api.py` — совместимый корневой wrapper для локального запуска из репозитория.

### `src/smart_scale/api`

- `app.py` — создание FastAPI-приложения, lifespan/startup warmup и exception handlers.
- `dependencies.py` — получение прогретого пайплайна из `app.state`.
- `schemas.py` — Pydantic-схемы ответа.
- `routes/predict.py` — endpoints `/api/predict` и `/api/health`.

### `src/smart_scale/ml`

- `anomaly.py` — MediaPipe Hands.
- `detection.py` — YOLO/localization и crop.
- `embedding.py` — DINOv2 + projection head / ONNX runtime.
- `vector_store.py` — `FaissVectorStore`, `PgVectorStore`, `CatalogIndexBuilder`.
- `pipeline.py` — оркестрация этапов инференса.

### `src/smart_scale/hardware`

- `scale.py` — чтение веса.
- `camera.py` — получение кадра.
- `controller.py` — стабилизация веса и захват payload для сервера.

## Хранилища и артефакты

- `images/` — локальный каталог изображений товаров и `product.csv`.
- `assets/models/` — веса текущей модели.
- `data/vector_db/` — локальный FAISS snapshot прототипа.

## Принципы

- API не знает деталей реализации моделей.
- Hardware не зависит от FastAPI.
- ML-пайплайн можно переключать между `file`, `faiss` и `pgvector`.
- FastAPI поднимает прогретый singleton `RecognitionPipeline` на startup и не обслуживает запросы до завершения warmup.
- Старые артефакты сохранены, но новый runtime код живёт только в `src/`.
- Проект упакован через `pyproject.toml`, поэтому runtime и тесты не должны зависеть от ручной модификации `sys.path`.
