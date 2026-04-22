# Архитектура Smart Scale

Документ описывает текущую версию проекта после перехода API на `assets/models/best.pt`.

## Целевой поток данных

1. Клиент отправляет изображение, `weight_grams` и `top_k` в FastAPI endpoint `/api/predict`.
2. `ml.anomaly.HandAnomalyDetector` проверяет кадр на руки.
3. Если `SMART_SCALE_PRODUCT_LOCALIZATION=1`, `ml.detection.ProductLocalizer` делает YOLO crop. В текущем default-режиме с `best.pt` этот шаг выключен.
4. `ml.embedding.DinoV2Embedder` строит embedding.
5. `ml.vector_store.PgVectorStore` выполняет KNN-поиск по `pgvector`.
6. `ml.pipeline.RecognitionPipeline` считает цену по весу и собирает доменный ответ.
7. `api/routes/predict.py` возвращает строгий JSON через Pydantic-схемы.

## Основной runtime

Текущий рекомендуемый production-like режим:

- encoder: `assets/models/best.pt`
- backbone внутри checkpoint: `vit_small_patch14_dinov2.lvd142m`
- embedding dim: `256`
- preprocessing: full-frame
- vector backend: `PostgreSQL + pgvector`
- table: `product_embeddings`
- hand detection: включена, если доступен `assets/models/hand_landmarker.task`
- product localization: выключена по умолчанию при наличии `best.pt`

`best.pt` загружается через `timm`. Head checkpoint должен совпадать с обучением: `Linear -> GELU -> Linear`. Выходной embedding нормализуется и сравнивается через cosine distance.

## Слои проекта

### Entry points

- `src/smart_scale/cli.py` - console entrypoint `smart-scale-api`.
- `src/smart_scale/bootstrap.py` - console entrypoint `smart-scale-bootstrap`.
- `src/smart_scale/__main__.py` - запуск через `python -m smart_scale`.
- `api.py` - корневой совместимый wrapper для запуска из репозитория.

### `src/smart_scale/api`

- `app.py` - создание FastAPI-приложения, lifespan/startup warmup, logging и exception handlers.
- `dependencies.py` - получение прогретого singleton `RecognitionPipeline` из `app.state`.
- `schemas.py` - Pydantic-схемы для health и predict.
- `routes/predict.py` - endpoints `/api/predict` и `/api/health`.
- `routes/ui.py` - вспомогательный UI/static слой, если он нужен при локальной работе.

### `src/smart_scale/ml`

- `anomaly.py` - MediaPipe Hands, блокировка инференса при руках в кадре.
- `detection.py` - YOLO-based crop с full-frame fallback.
- `embedding.py` - vanilla DINOv2 через `transformers` и fine-tuned timm checkpoint через `best.pt`.
- `catalog_seed.py` - построение каталога из PackEat-like dataset и прайс-листа.
- `vector_store.py` - `PgVectorStore`, `FaissVectorStore`, `FileVectorStore`.
- `pipeline.py` - orchestration: weight validation, hand check, optional localization, embedding, KNN, pricing.

### `src/smart_scale/hardware`

- `scale.py` - чтение веса.
- `camera.py` - получение кадра.
- `controller.py` - стабилизация веса и сбор payload для API.

## Embedding modes

### Fine-tuned checkpoint

Активируется автоматически, если существует `assets/models/best.pt`, или явно:

```powershell
$env:SMART_SCALE_EMBEDDING_CHECKPOINT = "$PWD\assets\models\best.pt"
$env:SMART_SCALE_EMBEDDING_DIM = "256"
```

Health должен показать:

```text
embedding_backend=timm_checkpoint:best.pt
```

### Vanilla DINOv2

Используется, если checkpoint отключен:

```powershell
$env:SMART_SCALE_EMBEDDING_CHECKPOINT = ""
$env:SMART_SCALE_EMBEDDING_DIM = "384"
```

Health покажет:

```text
embedding_backend=torch_transformers
```

## Product localization

Локализация разделена на два независимых флага:

- `SMART_SCALE_PRODUCT_LOCALIZATION` - crop на API inference.
- `SMART_SCALE_CATALOG_YOLO` - crop при bootstrap каталога.

Для корректного KNN train/catalog и test/API должны проходить через одинаковый preprocessing:

- full-frame + full-frame;
- или YOLO crop + YOLO crop.

Текущий default для `best.pt`:

```powershell
$env:SMART_SCALE_PRODUCT_LOCALIZATION = "0"
$env:SMART_SCALE_CATALOG_YOLO = "0"
```

В этом режиме `ProductLocalizer` не обязан быть готов, а health показывает `detector_name=full_frame_fallback`.

## PgVector schema

Текущая таблица:

```sql
CREATE TABLE product_embeddings (
  product_id TEXT PRIMARY KEY,
  embedding VECTOR(256) NOT NULL,
  product_type TEXT NOT NULL,
  product_sort TEXT NOT NULL,
  price_rub_per_kg DOUBLE PRECISION NOT NULL
);
```

Для vanilla DINOv2 таблица должна быть `VECTOR(384)`. Runtime API не пересоздает таблицу при mismatch, чтобы не потерять данные неожиданно. Bootstrap создает `PgVectorStore(..., recreate_on_dimension_mismatch=True)` и может пересоздать таблицу под текущую размерность.

## Bootstrap

`smart-scale-bootstrap` строит reference catalog:

1. Загружает settings.
2. Проверяет dataset и price catalog.
3. Создает `PgVectorStore`.
4. Прогревает encoder.
5. Опционально прогревает YOLO, если `SMART_SCALE_CATALOG_YOLO=1`.
6. Находит изображения сортов.
7. Берет `SMART_SCALE_SAMPLES_PER_SORT` изображений на сорт.
8. Строит embeddings batch-ами.
9. Атомарно заменяет каталог в `pgvector`.

Для текущего проекта рекомендуемый dataset path:

```powershell
$env:SMART_SCALE_DATASET_DIR = "$PWD\varieties_classification_dataset\train"
```

Так test split остается вне индекса и может использоваться для проверки accuracy.

## API startup

FastAPI lifespan:

1. Создает `RecognitionPipeline.from_settings`.
2. Проверяет runtime-компоненты.
3. Прогревает encoder.
4. Проверяет `pgvector` catalog count.
5. Запускает приложение только после успешного warmup.

В default-режиме `best.pt` startup не требует YOLO. Если включить `SMART_SCALE_PRODUCT_LOCALIZATION=1`, startup проверит готовность YOLO localizer.

## Health contract

`/api/health` возвращает:

- `status`
- `vector_backend`
- `vector_index_ready`
- `model_ready`
- `catalog_items`
- `warmup_completed`
- `hand_detection_enabled`
- `hand_detection_ready`
- `product_localization_enabled`
- `detection_model_ready`
- `detector_name`
- `embedding_backend`

Для текущего full-frame `best.pt` режима ожидаемо:

```json
{
  "vector_backend": "pgvector",
  "vector_index_ready": true,
  "model_ready": true,
  "catalog_items": 320,
  "product_localization_enabled": false,
  "detection_model_ready": false,
  "detector_name": "full_frame_fallback",
  "embedding_backend": "timm_checkpoint:best.pt"
}
```

## Predict contract

`POST /api/predict` принимает multipart form:

- `image`: файл изображения;
- `weight_grams`: положительное число;
- `top_k`: число результатов, default `3`.

Ответ содержит:

- статус `ok`, `warning` или `error`;
- лучший `product`;
- `top_matches`;
- `total_price`;
- `crop`;
- `embedding_dim`;
- `pipeline_steps`.

Если вес меньше или равен нулю, pipeline возвращает business error. Если руки обнаружены, pipeline возвращает warning и не запускает embedding/KNN.

## Принципы

- API не знает деталей реализации моделей и работает через `RecognitionPipeline`.
- Hardware слой не зависит от FastAPI.
- Vector store можно переключать между `pgvector`, `faiss` и `file`.
- `pgvector` является текущим основным backend.
- Runtime не должен молча смешивать 256d и 384d embeddings.
- Preprocessing каталога и API должен быть одинаковым.
- Старые локальные/FAISS режимы сохранены как fallback и для тестов.
