# Smart Scale Recognition Service

Распознавание фруктов и овощей по фото и весу. Пайплайн: MediaPipe Hands для блокировки кадров с руками, `yolo11n-seg` для сегментации объекта, `DinoV2`-эмбеддинги для поиска по локальному каталогу.

## Структура проекта

```text
.
|-- api.py                                # Совместимый thin wrapper для локального запуска
|-- assets/
|   `-- models/                           # ONNX/PyTorch, YOLO и MediaPipe assets
|-- data/
|   `-- vector_db/                        # Файловый векторный индекс
|-- docs/
|   |-- TASK.MD
|   `-- architecture.md
|-- images/                               # Локальный каталог изображений + product.csv
|-- pyproject.toml                        # Метаданные пакета и console script
|-- src/
|   `-- smart_scale/
|       |-- api/
|       |   |-- app.py                    # Lifespan, startup warmup, root endpoint
|       |   |-- dependencies.py
|       |   |-- errors.py
|       |   |-- schemas.py
|       |   `-- routes/
|       |       `-- predict.py            # /api/predict и /api/health
|       |-- cli.py                        # Console entrypoint smart-scale-api
|       |-- config.py                     # Settings и env vars
|       |-- domain/
|       |   `-- models.py
|       |-- hardware/
|       |   |-- camera.py
|       |   |-- controller.py
|       |   `-- scale.py
|       `-- ml/
|           |-- anomaly.py                # MediaPipe Hands
|           |-- detection.py              # YOLO segmentation + crop
|           |-- embedding.py              # DinoV2 embedder
|           |-- pipeline.py               # RecognitionPipeline
|           `-- vector_store.py           # File/FAISS/PgVector backends
|-- tests/
|   |-- _bootstrap.py                     # Общий bootstrap для test imports
|   |-- test_api.py
|   `-- test_ml_pipeline.py
`-- requirements.txt
```

## Установка

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

`requirements.txt` остаётся источником runtime-зависимостей, а `pyproject.toml` делает проект устанавливаемым в editable-режиме без ручной правки `PYTHONPATH`.

Если активация окружения запрещена политикой PowerShell, можно использовать `.venv\Scripts\python.exe` напрямую:

```powershell
.venv\Scripts\python.exe -m pip install -e .
```

## Модели и данные

Обязательные файлы в `assets/models/`:

- `fruit_embedder_final.onnx`
- `fruit_embedder_final.onnx.data`
- `fruit_embedder_final.pth`
- `yolo11n-seg.pt`

Опционально:

- `hand_landmarker.task`

Каталог для поиска должен быть доступен одним из двух способов:

- готовый индекс `data/vector_db/catalog.pkl`
- или исходные данные `images/` + `images/product.csv`

Если каких-то моделей нет, их можно скачать вручную:

```powershell
Invoke-WebRequest -UseBasicParsing `
  -Uri "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-seg.pt" `
  -OutFile "assets/models/yolo11n-seg.pt"

Invoke-WebRequest -UseBasicParsing `
  -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
  -OutFile "assets/models/hand_landmarker.task"
```

## Запуск

Рекомендуемый способ после `pip install -e .`:

```powershell
smart-scale-api
```

Альтернатива через пакет:

```powershell
.venv\Scripts\python.exe -m smart_scale
```

Совместимый запуск через корневой wrapper:

```powershell
.venv\Scripts\python.exe api.py
```

Локально через uvicorn:

```powershell
.venv\Scripts\python.exe -m uvicorn smart_scale.api:app --host 0.0.0.0 --port 8000
```

Сервис на старте делает eager warmup:

1. валидирует конфиг и пути к моделям;
2. создаёт `RecognitionPipeline`;
3. прогревает embedder;
4. инициализирует hand detector и YOLO;
5. загружает файловый индекс или собирает его из `images/`.

Если критический компонент не готов, API не поднимается частично, а падает на старте.

Остановка сервера:

```powershell
Ctrl+C
```

Если процесс висит в фоне:

```powershell
cmd /c netstat -ano | findstr :8000
Stop-Process -Id <PID>
```

## URL

| URL | Описание |
|-----|----------|
| http://localhost:8000/ | Root endpoint с ссылками на API |
| http://localhost:8000/docs | Swagger UI |
| http://localhost:8000/api/health | Readiness после warmup |

## API

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | Короткий discovery endpoint |
| GET | `/api/health` | Статус прогрева и готовности пайплайна |
| POST | `/api/predict` | Фото + вес -> распознавание товара |

**POST /api/predict**: `image`, `weight_grams`, `top_k`

Входной формат: `multipart/form-data`

Пример PowerShell:

```powershell
curl.exe -X POST `
  -F "image=@images/r0_10.jpg" `
  -F "weight_grams=100" `
  -F "top_k=3" `
  http://127.0.0.1:8000/api/predict
```

Пример ответа:

```json
{
  "status": "ok",
  "message": "Product recognized.",
  "weight_grams": 100.0,
  "total_price": 10000.0,
  "product": {
    "product_id": "r0_10/r0_10",
    "name": "r0_10",
    "score": 0.9826,
    "price_per_gram": 100.0
  },
  "top_matches": [],
  "anomaly_detected": false,
  "warning_code": null,
  "crop": {
    "bbox": [0, 4, 453, 383],
    "confidence": 0.5289,
    "detector_name": "yolo11n-seg.pt",
    "mask_applied": true
  },
  "embedding_dim": 256
}
```

Семантика `status`:

- `ok` — товар найден
- `warning` — бизнес-блокировка, например обнаружены руки
- `error` — запрос валиден, но результат распознавания не получен

HTTP-коды:

- `200` — валидный запрос, итог смотреть в поле `status`
- `400` — битое изображение, пустой файл или невалидные form fields
- `503` — сервис не готов
- `500` — внутренняя ошибка

## Тесты

```powershell
.venv\Scripts\python.exe -m unittest tests.test_ml_pipeline -v
.venv\Scripts\python.exe -m unittest tests.test_api -v
```

Или полный прогон:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

## Переменные окружения

| Переменная | Описание |
|------------|----------|
| `SMART_SCALE_API_HOST` | Хост FastAPI, по умолчанию `0.0.0.0` |
| `SMART_SCALE_API_PORT` | Порт API, по умолчанию `8000` |
| `SMART_SCALE_API_TITLE` | Имя сервиса в Swagger и root endpoint |
| `SMART_SCALE_MODEL_PATH` | Путь до PyTorch checkpoint embedder |
| `SMART_SCALE_ONNX_PATH` | Путь до ONNX-модели embedder |
| `SMART_SCALE_DETECTION_MODEL` | Путь до `yolo11n-seg.pt` |
| `SMART_SCALE_HAND_LANDMARKER_PATH` | Путь до `hand_landmarker.task` |
| `SMART_SCALE_HAND_DETECTION` | Включает или выключает hand detection |
| `SMART_SCALE_VECTOR_BACKEND` | Backend поиска: `file`, `faiss`, `pgvector` |
| `SMART_SCALE_FILE_VECTOR_STORE_PATH` | Путь до файлового индекса |
| `SMART_SCALE_IMAGE_DIR` | Папка с изображениями каталога |
| `SMART_SCALE_PRODUCTS_CSV` | CSV с метаданными каталога |
| `SMART_SCALE_TOP_K` | Значение `top_k` по умолчанию |
| `SMART_SCALE_PRICE_PRECISION` | Точность расчёта цены |

## Отладка

- Если startup падает с `PytorchStreamReader failed reading zip archive`, файл `yolo11n-seg.pt` повреждён или обрезан.
- Если `GET /api/health` не поднимается, проверь пути к ONNX/PyTorch asset и наличие локального индекса.
- Если `hand_landmarker.task` отсутствует, сервис может стартовать, но hand stage будет работать в skip-режиме.
- Если порт `8000` занят, выстави другой порт через `SMART_SCALE_API_PORT`.

## Дополнительно

- [docs/TASK.MD](docs/TASK.MD)
- [docs/architecture.md](docs/architecture.md)
