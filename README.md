# Smart Scale

Сервис распознавания фруктов и овощей по изображению и весу.

Текущий пайплайн:
- `MediaPipe Hands` для блокировки кадров с руками
- `YOLO11n-seg` для локализации и сегментации продукта
- `DINOv2` для построения эмбеддингов
- `PostgreSQL + pgvector` для поиска ближайшего товара по эмбеддингу

Проект уже настроен под каталог продуктов в `pgvector` со схемой:
- `product_id`
- `embedding`
- `product_type`
- `product_sort`
- `price_rub_per_kg`

## Что нужно для работы

- Windows + PowerShell
- Python `3.11+`
- Docker Desktop
- локальный датасет в [varieties_classification_dataset](varieties_classification_dataset)
- прайс-лист в [product_prices.py](data/product_prices.py)
- модели в [assets/models](assets/models)

Обязательные файлы моделей:
- `assets/models/fruit_embedder_final.onnx`
- `assets/models/fruit_embedder_final.onnx.data`
- `assets/models/fruit_embedder_final.pth`
- `assets/models/yolo11n-seg.pt`

Опционально:
- `assets/models/hand_landmarker.task`

Если `hand_landmarker.task` отсутствует, API всё равно запустится, но этап проверки рук будет работать в skip-режиме.

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
|-- src/
|   `-- smart_scale/
|       |-- api/
|       |-- bootstrap.py
|       |-- cli.py
|       |-- config.py
|       |-- domain/
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

## Конфигурация по умолчанию

Сейчас проект ожидает такие default-значения:

- `SMART_SCALE_VECTOR_BACKEND=pgvector`
- `SMART_SCALE_PGVECTOR_DSN=postgresql://smart_scale:smart_scale@localhost:5433/smart_scale`
- `SMART_SCALE_PGVECTOR_TABLE=product_embeddings`
- `SMART_SCALE_DATASET_DIR=D:\Yandex_Cemp_Smart_Scale\varieties_classification_dataset`
- `SMART_SCALE_PRICE_CATALOG=D:\Yandex_Cemp_Smart_Scale\data\product_prices.py`
- `SMART_SCALE_SAMPLES_PER_SORT=5`
- `SMART_SCALE_API_HOST=0.0.0.0`
- `SMART_SCALE_API_PORT=8000`

То есть базовый happy path уже рассчитан на локальный `pgvector` на порту `5433`.

## Установка зависимостей

Рекомендуемый вариант:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Если активация PowerShell запрещена политикой:

```powershell
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e .
```

`pyproject.toml` уже регистрирует console scripts:
- `smart-scale-api`
- `smart-scale-bootstrap`

## Как поднять БД

В проекте есть готовый compose для `pgvector`.

Запуск:

```powershell
docker compose -f docker-compose.pgvector.yml up -d
```

Что поднимется:
- контейнер `smart-scale-pgvector`
- PostgreSQL с расширением `pgvector`
- внешний порт `5433`

Строка подключения:

```text
postgresql://smart_scale:smart_scale@localhost:5433/smart_scale
```

Проверка, что база жива:

```powershell
docker ps
docker exec smart-scale-pgvector pg_isready -U smart_scale -d smart_scale
```

Проверка, что таблица уже заполнена:

```powershell
docker exec smart-scale-pgvector psql -U smart_scale -d smart_scale -c "SELECT COUNT(*) FROM product_embeddings;"
```

## Как заполнить БД

Рекомендуемый сценарий: заполнять БД отдельной bootstrap-командой, а не надеяться на автосборку при старте API.

### 1. Выставить переменные окружения

```powershell
$env:SMART_SCALE_VECTOR_BACKEND = "pgvector"
$env:SMART_SCALE_PGVECTOR_DSN = "postgresql://smart_scale:smart_scale@localhost:5433/smart_scale"
$env:SMART_SCALE_PGVECTOR_TABLE = "product_embeddings"
$env:SMART_SCALE_DATASET_DIR = "D:\Yandex_Cemp_Smart_Scale\varieties_classification_dataset"
$env:SMART_SCALE_PRICE_CATALOG = "D:\Yandex_Cemp_Smart_Scale\data\product_prices.py"
$env:SMART_SCALE_SAMPLES_PER_SORT = "5"
```

### 2. Запустить bootstrap

```powershell
smart-scale-bootstrap
```

Или без активации окружения:

```powershell
.venv\Scripts\python.exe -m smart_scale.bootstrap
```

Что делает bootstrap:
- прогревает embedder
- создаёт таблицу `product_embeddings`, если её ещё нет
- мигрирует старую схему, если это возможно
- находит изображения сортов в `varieties_classification_dataset`
- берёт по `5` изображений на каждый сорт из прайс-листа
- строит эмбеддинги
- атомарно заменяет каталог в `pgvector`

Ожидаемый результат для текущего проекта:
- `64` сортов в [product_prices.py](data/product_prices.py)
- `5` эмбеддингов на сорт
- всего `320` строк в `product_embeddings`

Быстрая проверка после bootstrap:

```powershell
docker exec smart-scale-pgvector psql -U smart_scale -d smart_scale -c "SELECT COUNT(*) FROM product_embeddings;"
```

## Как запустить API

После того как БД уже заполнена:

```powershell
$env:SMART_SCALE_VECTOR_BACKEND = "pgvector"
$env:SMART_SCALE_PGVECTOR_DSN = "postgresql://smart_scale:smart_scale@localhost:5433/smart_scale"
$env:SMART_SCALE_BUILD_INDEX = "0"
smart-scale-api
```

Альтернативы:

```powershell
.venv\Scripts\python.exe -m smart_scale
```

или

```powershell
.venv\Scripts\python.exe api.py
```

На старте сервис делает warmup:
- проверяет модели и конфиг
- инициализирует YOLO
- инициализирует MediaPipe
- прогревает embedder
- проверяет доступность каталога в `pgvector`

По умолчанию API поднимается на:

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/api/health`

## Как запустить API с автоматическим bootstrap

Так можно, но это менее удобно для отладки.

```powershell
$env:SMART_SCALE_VECTOR_BACKEND = "pgvector"
$env:SMART_SCALE_PGVECTOR_DSN = "postgresql://smart_scale:smart_scale@localhost:5433/smart_scale"
$env:SMART_SCALE_BUILD_INDEX = "1"
smart-scale-api
```

В этом режиме, если таблица пуста, warmup сам попробует собрать каталог из:
- `SMART_SCALE_DATASET_DIR`
- `SMART_SCALE_PRICE_CATALOG`

Для рабочей эксплуатации всё равно лучше отдельный `smart-scale-bootstrap`, а API держать с `SMART_SCALE_BUILD_INDEX=0`.

## Как проверить, что сервис реально работает

### Healthcheck

```powershell
curl.exe http://127.0.0.1:8000/api/health
```

Ожидаемый смысл полей:
- `vector_backend=pgvector`
- `vector_index_ready=true`
- `catalog_items=320`
- `warmup_completed=true`

### Swagger

Открыть:

```text
http://127.0.0.1:8000/docs
```

### Пример запроса на распознавание

```powershell
curl.exe -X POST `
  -F "image=@images/r0_10.jpg" `
  -F "weight_grams=100" `
  -F "top_k=3" `
  http://127.0.0.1:8000/api/predict
```

Пример актуального ответа:

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
    "score": 0.98,
    "price_rub_per_kg": 175.0,
    "metadata": {
      "product_type": "apple",
      "product_sort": "fuji",
      "price_rub_per_kg": 175.0
    }
  },
  "top_matches": [],
  "anomaly_detected": false,
  "warning_code": null,
  "crop": {
    "bbox": [0, 0, 100, 100],
    "confidence": 0.95,
    "detector_name": "yolo11n-seg.pt",
    "mask_applied": true
  },
  "embedding_dim": 256,
  "pipeline_steps": [
    "weight_received",
    "anomaly_check_started",
    "anomaly_check_completed",
    "localization_completed",
    "embedding_completed",
    "knn_search_completed",
    "response_ready"
  ]
}
```

## Тесты

Локально:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

Через Docker:

```powershell
docker compose -f docker-compose.tests.yml up --build --abort-on-container-exit --exit-code-from tests
```

## Полезные переменные окружения

Основные:

- `SMART_SCALE_VECTOR_BACKEND`
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

Пути до моделей:

- `SMART_SCALE_MODEL_PATH`
- `SMART_SCALE_ONNX_PATH`
- `SMART_SCALE_DETECTION_MODEL`
- `SMART_SCALE_HAND_LANDMARKER_PATH`

Поведение hand detection:

- `SMART_SCALE_HAND_DETECTION`

## Рекомендуемый сценарий "с нуля"

Если нужно поднять проект полностью с чистой машины:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
docker compose -f docker-compose.pgvector.yml up -d

$env:SMART_SCALE_VECTOR_BACKEND = "pgvector"
$env:SMART_SCALE_PGVECTOR_DSN = "postgresql://smart_scale:smart_scale@localhost:5433/smart_scale"
$env:SMART_SCALE_PGVECTOR_TABLE = "product_embeddings"
$env:SMART_SCALE_DATASET_DIR = "D:\Yandex_Cemp_Smart_Scale\varieties_classification_dataset"
$env:SMART_SCALE_PRICE_CATALOG = "D:\Yandex_Cemp_Smart_Scale\data\product_prices.py"
$env:SMART_SCALE_SAMPLES_PER_SORT = "5"
$env:SMART_SCALE_BUILD_INDEX = "0"

smart-scale-bootstrap
smart-scale-api
```

После этого:
- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/api/health`

## Отладка

- Если `smart-scale-bootstrap` падает с ошибкой по БД, сначала проверь `docker exec smart-scale-pgvector pg_isready -U smart_scale -d smart_scale`.
- Если сервис не стартует, проверь наличие моделей в `assets/models/`.
- Если `catalog_items=0`, значит БД не заполнена или указан не тот `SMART_SCALE_PGVECTOR_DSN`.
- Если при первом старте embedder тянет Hugging Face cache, это ожидаемо: `AutoImageProcessor` для `facebook/dinov2-small` может скачать служебные файлы в кэш.
- Если порт `8000` занят, задай другой через `SMART_SCALE_API_PORT`.
- Если порт `5433` уже занят, поменяй маппинг в [docker-compose.pgvector.yml](docker-compose.pgvector.yml) и такой же DSN в окружении.

## Дополнительно

- [TASK.MD](docs/TASK.MD)
- [architecture.md](docs/architecture.md)
- [pgvector_bootstrap.md](docs/pgvector_bootstrap.md)
