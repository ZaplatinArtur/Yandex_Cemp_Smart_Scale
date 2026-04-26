# Smart Scale

Сервис распознавания фруктов и овощей по изображению и весу.

Текущий рабочий сценарий:

- FastAPI принимает изображение и вес через `/api/predict`.
- MediaPipe Hands блокирует инференс, если в кадре найдены руки.
- По умолчанию изображение идет в encoder целиком, без YOLO crop.
- Fine-tuned `assets/models/best.pt` строит 256-мерный embedding.
- PostgreSQL + pgvector ищет ближайшие товары по cosine distance.
- Ответ содержит top-k совпадений, цену за кг и итоговую цену по весу.

`best.pt` - это не обычный классификатор. Модель работает как encoder: она превращает изображение в нормализованный embedding, а класс товара определяется через KNN-поиск по каталогу в `pgvector`.

## Текущий default

Если файл `assets/models/best.pt` существует, проект автоматически использует его:

- embedding backend: `timm_checkpoint:best.pt`
- embedding dim: `256`
- product localization: `false`
- crop mode: `full_frame`
- vector backend: `pgvector`
- table: `product_embeddings`

Если `best.pt` отключить или удалить, проект откатится к vanilla `facebook/dinov2-small` с 384-мерными embedding:

```powershell
$env:SMART_SCALE_EMBEDDING_CHECKPOINT = ""
$env:SMART_SCALE_EMBEDDING_DIM = "384"
```

## Что нужно для работы

- Windows + PowerShell
- Python `3.11+`
- Docker Desktop
- датасет `varieties_classification_dataset`
- прайс-лист `data/product_prices.py`
- модели в `assets/models`

Рекомендуемые артефакты:

- `assets/models/best.pt` - текущий encoder для API и метрик
- `assets/models/hand_landmarker.task` - проверка рук через MediaPipe
- `assets/models/yolo_int8.onnx` или `assets/models/yolo.onnx` - опциональная YOLO-локализация
- `assets/models/yolo11n-seg.pt` - исходная YOLO-seg модель для экспериментов и экспорта

Если `hand_landmarker.task` отсутствует, API запустится, но проверка рук будет пропущена.

## Скачивание датасета

Датасет `varieties_classification_dataset` содержит изображения 64 сортов фруктов и овощей, разделенные на train и test выборки.

**Скачать датасет:**

https://drive.google.com/file/d/1KYP3B15wL_P5HSUwMIuLgF47NrzD0X4s/view?usp=sharing

После скачивания распакуйте архив в корень проекта так, чтобы получилась структура:

```text
varieties_classification_dataset/
├── train/
│   ├── apple_fuji/
│   ├── apple_golden/
│   └── ...
└── test/
    ├── apple_fuji/
    ├── apple_golden/
    └── ...
```

## Структура проекта

```text
.
|-- api.py
|-- assets/
|   `-- models/
|-- data/
|   `-- product_prices.py
|-- docs/
|-- images/
|-- scripts/
|-- src/
|   `-- smart_scale/
|       |-- api/
|       |-- bootstrap.py
|       |-- cli.py
|       |-- config.py
|       |-- domain/
|       |-- hardware/
|       `-- ml/
|-- tests/
|-- varieties_classification_dataset/
|   |-- train/
|   `-- test/
|-- docker-compose.pgvector.yml
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Установка

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Если PowerShell запрещает активацию окружения:

```powershell
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e .
```

После установки доступны команды:

- `smart-scale-api`
- `smart-scale-bootstrap`

## Поднять pgvector

```powershell
docker compose -f docker-compose.pgvector.yml up -d
```

Контейнер:

- name: `smart-scale-pgvector`
- database: `smart_scale`
- user/password: `smart_scale` / `smart_scale`
- external port: `5433`

DSN по умолчанию:

```text
postgresql://smart_scale:smart_scale@localhost:5433/smart_scale
```

Проверка:

```powershell
docker exec smart-scale-pgvector pg_isready -U smart_scale -d smart_scale
```

## Заполнить БД из train

Для API-режима и честной проверки метрик каталог нужно строить из `train`, а тестовые изображения держать вне индекса.

```powershell
$env:SMART_SCALE_VECTOR_BACKEND = "pgvector"
$env:SMART_SCALE_PGVECTOR_DSN = "postgresql://smart_scale:smart_scale@localhost:5433/smart_scale"
$env:SMART_SCALE_PGVECTOR_TABLE = "product_embeddings"
$env:SMART_SCALE_DATASET_DIR = "$PWD\varieties_classification_dataset\train"
$env:SMART_SCALE_PRICE_CATALOG = "$PWD\data\product_prices.py"
$env:SMART_SCALE_EMBEDDING_CHECKPOINT = "$PWD\assets\models\best.pt"
$env:SMART_SCALE_EMBEDDING_DIM = "256"
$env:SMART_SCALE_SAMPLES_PER_SORT = "5"
$env:SMART_SCALE_CATALOG_YOLO = "0"
$env:SMART_SCALE_PRODUCT_LOCALIZATION = "0"

smart-scale-bootstrap
```

Без активированного окружения:

```powershell
.venv\Scripts\python.exe -m smart_scale.bootstrap
```

Что делает bootstrap:

- прогревает encoder;
- создает extension `vector`;
- создает или мигрирует таблицу `product_embeddings`;
- при несовпадении размерности таблицы пересоздает ее под текущий embedding dim;
- берет по `5` изображений на сорт из `train`;
- строит embedding через `best.pt`;
- атомарно заменяет каталог в `pgvector`.

Ожидаемый результат для текущего прайс-листа:

- `64` сорта из `data/product_prices.py`
- `5` embedding на сорт
- `320` строк в `product_embeddings`
- `vector(256)`

Проверка:

```powershell
docker exec smart-scale-pgvector psql -U smart_scale -d smart_scale -c "SELECT COUNT(*), vector_dims(embedding) FROM product_embeddings GROUP BY vector_dims(embedding);"
```

Ожидаемо:

```text
 count | vector_dims
-------+-------------
   320 |         256
```

## Запустить API

```powershell
$env:SMART_SCALE_VECTOR_BACKEND = "pgvector"
$env:SMART_SCALE_PGVECTOR_DSN = "postgresql://smart_scale:smart_scale@localhost:5433/smart_scale"
$env:SMART_SCALE_PGVECTOR_TABLE = "product_embeddings"
$env:SMART_SCALE_EMBEDDING_CHECKPOINT = "$PWD\assets\models\best.pt"
$env:SMART_SCALE_EMBEDDING_DIM = "256"
$env:SMART_SCALE_PRODUCT_LOCALIZATION = "0"
$env:SMART_SCALE_BUILD_INDEX = "0"

smart-scale-api
```

Альтернативы:

```powershell
.venv\Scripts\python.exe -m smart_scale
```

```powershell
.venv\Scripts\python.exe api.py
```

URL:

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/api/health`

На startup API делает warmup: проверяет конфиг, прогревает encoder, подключается к `pgvector` и проверяет, что каталог не пустой. В default-режиме с `best.pt` YOLO не загружается.

## Проверить API

Healthcheck:

```powershell
curl.exe http://127.0.0.1:8000/api/health
```

Смысл ключевых полей:

- `vector_backend`: должен быть `pgvector`
- `vector_index_ready`: `true`
- `catalog_items`: `320`
- `model_ready`: `true`
- `embedding_backend`: `timm_checkpoint:best.pt`
- `product_localization_enabled`: `false`
- `detector_name`: `full_frame_fallback`

Пример запроса:

```powershell
curl.exe -X POST `
  -F "weight_grams=100" `
  -F "top_k=5" `
  -F "image=@varieties_classification_dataset/test/apple_fuji/001222.jpg;type=image/jpeg" `
  http://127.0.0.1:8000/api/predict
```

Типичный ответ:

```json
{
  "status": "ok",
  "message": "Товар распознан.",
  "weight_grams": 100.0,
  "total_price": 17.5,
  "product": {
    "product_id": "apple_fuji:01",
    "name": "apple_fuji",
    "product_type": "apple",
    "product_sort": "fuji",
    "score": 0.9385,
    "price_rub_per_kg": 175.0
  },
  "crop": {
    "bbox": [0, 0, 1440, 1080],
    "confidence": 1.0,
    "detector_name": "full_frame",
    "mask_applied": false
  },
  "embedding_dim": 256,
  "pipeline_steps": [
    "weight_received",
    "anomaly_check_started",
    "anomaly_check_completed",
    "localization_skipped",
    "embedding_completed",
    "knn_search_completed",
    "response_ready"
  ]
}
```

## YOLO-режим

YOLO сейчас опционален. Он нужен только если вы хотите обрезать продукт перед encoder.

Включить YOLO crop для API:

```powershell
$env:SMART_SCALE_PRODUCT_LOCALIZATION = "1"
```

Включить YOLO crop при bootstrap каталога:

```powershell
$env:SMART_SCALE_CATALOG_YOLO = "1"
```

Важно: train и test должны проходить через один и тот же preprocessing. Если каталог построен full-frame, API тоже должен работать full-frame. Если каталог построен через YOLO crop, API тоже должен использовать YOLO crop. Иначе embeddings сравниваются в разных распределениях, и accuracy падает.

## Конфигурация

Основные переменные:

- `SMART_SCALE_VECTOR_BACKEND` - `pgvector`, `faiss` или `file`
- `SMART_SCALE_PGVECTOR_DSN`
- `SMART_SCALE_PGVECTOR_TABLE`
- `SMART_SCALE_DATASET_DIR`
- `SMART_SCALE_PRICE_CATALOG`
- `SMART_SCALE_SAMPLES_PER_SORT`
- `SMART_SCALE_BUILD_INDEX`
- `SMART_SCALE_API_HOST`
- `SMART_SCALE_API_PORT`
- `SMART_SCALE_TOP_K`
- `SMART_SCALE_PRICE_PRECISION`
- `SMART_SCALE_EMBEDDING_MODEL_NAME`
- `SMART_SCALE_EMBEDDING_CHECKPOINT`
- `SMART_SCALE_EMBEDDING_DIM`
- `SMART_SCALE_PRODUCT_LOCALIZATION`
- `SMART_SCALE_CATALOG_YOLO`
- `SMART_SCALE_DETECTION_MODEL`
- `SMART_SCALE_DETECTION_QUANT_PATH`
- `SMART_SCALE_HAND_LANDMARKER_PATH`
- `SMART_SCALE_HAND_DETECTION`

Defaults:

- `SMART_SCALE_VECTOR_BACKEND=pgvector`
- `SMART_SCALE_PGVECTOR_DSN=postgresql://smart_scale:smart_scale@localhost:5433/smart_scale`
- `SMART_SCALE_PGVECTOR_TABLE=product_embeddings`
- `SMART_SCALE_DATASET_DIR=<project>\varieties_classification_dataset`
- `SMART_SCALE_PRICE_CATALOG=<project>\data\product_prices.py`
- `SMART_SCALE_SAMPLES_PER_SORT=5`
- `SMART_SCALE_API_HOST=0.0.0.0`
- `SMART_SCALE_API_PORT=8000`

Для bootstrap без утечки test-выборки рекомендуется явно задавать:

```powershell
$env:SMART_SCALE_DATASET_DIR = "$PWD\varieties_classification_dataset\train"
```

## Метрики текущего encoder

Последний локальный offline-прогон для `assets/models/best.pt`:

- protocol: train catalog, `k=5` samples per sort, full-frame
- embedding dim: `256`
- top1 accuracy: `0.8849`
- top3 accuracy: `0.9441`
- top5 accuracy: `0.9572`

Если API показывает заметно меньше, сначала проверьте:

- таблица действительно `vector(256)`;
- `embedding_backend=timm_checkpoint:best.pt`;
- `product_localization_enabled=false`;
- каталог построен из `train`, а проверка идет по `test`;
- в API и bootstrap одинаковые `SMART_SCALE_PRODUCT_LOCALIZATION` / `SMART_SCALE_CATALOG_YOLO`.

## Тесты

Локально:

```powershell
$env:PYTHONPATH = "$PWD\src"
.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

Через Docker:

```powershell
docker compose -f docker-compose.tests.yml up --build --abort-on-container-exit --exit-code-from tests
```

## Сценарий с нуля

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .

docker compose -f docker-compose.pgvector.yml up -d

$env:SMART_SCALE_VECTOR_BACKEND = "pgvector"
$env:SMART_SCALE_PGVECTOR_DSN = "postgresql://smart_scale:smart_scale@localhost:5433/smart_scale"
$env:SMART_SCALE_PGVECTOR_TABLE = "product_embeddings"
$env:SMART_SCALE_DATASET_DIR = "$PWD\varieties_classification_dataset\train"
$env:SMART_SCALE_PRICE_CATALOG = "$PWD\data\product_prices.py"
$env:SMART_SCALE_EMBEDDING_CHECKPOINT = "$PWD\assets\models\best.pt"
$env:SMART_SCALE_EMBEDDING_DIM = "256"
$env:SMART_SCALE_SAMPLES_PER_SORT = "5"
$env:SMART_SCALE_CATALOG_YOLO = "0"
$env:SMART_SCALE_PRODUCT_LOCALIZATION = "0"
$env:SMART_SCALE_BUILD_INDEX = "0"

smart-scale-bootstrap
smart-scale-api
```

## Отладка

- Если bootstrap падает по БД, проверьте `docker exec smart-scale-pgvector pg_isready -U smart_scale -d smart_scale`.
- Если API падает с mismatch размерности, пересоберите `product_embeddings` через `smart-scale-bootstrap`.
- Если `catalog_items=0`, таблица пуста или указан не тот DSN/table.
- Если `embedding_backend` не `timm_checkpoint:best.pt`, проверьте `SMART_SCALE_EMBEDDING_CHECKPOINT`.
- Если `embedding_dim` не `256`, проверьте `SMART_SCALE_EMBEDDING_DIM`.
- Если accuracy падает до уровня около `0.5`, проверьте, не включен ли YOLO crop только на одной стороне: API или catalog bootstrap.
- Если порт `8000` занят, задайте `SMART_SCALE_API_PORT`.
- Если порт `5433` занят, измените mapping в `docker-compose.pgvector.yml` и DSN.

## Дополнительно

- [docs/architecture.md](docs/architecture.md)
- [docs/pgvector_bootstrap.md](docs/pgvector_bootstrap.md)
