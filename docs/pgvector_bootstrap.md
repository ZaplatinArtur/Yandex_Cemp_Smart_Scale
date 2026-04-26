# PgVector Bootstrap

Инструкция для заполнения `PostgreSQL + pgvector` каталога текущими embeddings.

Текущий рекомендуемый каталог:

- source split: `varieties_classification_dataset/train`
- encoder: `assets/models/best.pt`
- embedding dim: `256`
- preprocessing: full-frame
- samples per sort: `5`
- expected rows: `320`

## Поднять БД

```powershell
docker compose -f docker-compose.pgvector.yml up -d
```

DSN:

```text
postgresql://smart_scale:smart_scale@localhost:5433/smart_scale
```

Проверка:

```powershell
docker exec smart-scale-pgvector pg_isready -U smart_scale -d smart_scale
```

## Переменные окружения

Рекомендуемый full-frame bootstrap через `best.pt`:

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
```

`SMART_SCALE_PRODUCT_LOCALIZATION` влияет на API inference. Для bootstrap каталога используется `SMART_SCALE_CATALOG_YOLO`.

## Запуск

```powershell
smart-scale-bootstrap
```

Или без активированного окружения:

```powershell
.venv\Scripts\python.exe -m smart_scale.bootstrap
```

Успешный запуск для текущего проекта:

```text
Bootstrapped 320 embeddings into product_embeddings using 5 samples per sort and catalog_yolo=False.
```

## Что делает bootstrap

1. Проверяет `SMART_SCALE_PGVECTOR_DSN`.
2. Проверяет dataset и price catalog.
3. Создает extension `vector`.
4. Создает таблицу `product_embeddings`, если ее нет.
5. Проверяет размерность существующей таблицы.
6. Если размерность не совпадает с текущим encoder, пересоздает таблицу.
7. Загружает `best.pt` и строит batch embeddings.
8. Записывает `product_id`, `embedding`, `product_type`, `product_sort`, `price_rub_per_kg`.
9. Заменяет каталог атомарно через staging table.

Важно: пересоздание таблицы при mismatch размерности происходит только в bootstrap. API runtime при mismatch падает с ошибкой, чтобы не удалить каталог во время обслуживания запросов.

## Проверка результата

Количество строк и размерность:

```powershell
docker exec smart-scale-pgvector psql -U smart_scale -d smart_scale -c "SELECT COUNT(*), vector_dims(embedding) FROM product_embeddings GROUP BY vector_dims(embedding);"
```

Ожидаемо:

```text
 count | vector_dims
-------+-------------
   320 |         256
```

Схема:

```powershell
docker exec smart-scale-pgvector psql -U smart_scale -d smart_scale -c "\d product_embeddings"
```

Краткая проверка первых строк:

```powershell
docker exec smart-scale-pgvector psql -U smart_scale -d smart_scale -c "SELECT product_id, product_type, product_sort, price_rub_per_kg FROM product_embeddings ORDER BY product_id LIMIT 10;"
```

## YOLO bootstrap

Если нужно строить каталог по YOLO crop, включите:

```powershell
$env:SMART_SCALE_CATALOG_YOLO = "1"
```

Для корректной работы API после этого нужно включить такой же preprocessing:

```powershell
$env:SMART_SCALE_PRODUCT_LOCALIZATION = "1"
```

Не смешивайте full-frame catalog с YOLO-cropped API и наоборот. В этом случае embeddings сравниваются по разным распределениям, что обычно ухудшает accuracy.

## Vanilla DINOv2 fallback

Если нужно пересобрать каталог без `best.pt`:

```powershell
$env:SMART_SCALE_EMBEDDING_CHECKPOINT = ""
$env:SMART_SCALE_EMBEDDING_DIM = "384"
smart-scale-bootstrap
```

После этого таблица будет `vector(384)`. API должен запускаться с теми же переменными.

## Частые ошибки

- `catalog_items=0`: bootstrap не запускался, таблица другая или указан неверный DSN.
- `Existing pgvector table ... vector(384), but current model produces vector(256)`: пересоберите через `smart-scale-bootstrap`.
- `Price catalog not found`: проверьте `SMART_SCALE_PRICE_CATALOG`.
- `Dataset directory not found`: проверьте `SMART_SCALE_DATASET_DIR`.
- Много ошибок классификации в API: проверьте, что API и bootstrap используют одинаковые `SMART_SCALE_PRODUCT_LOCALIZATION` / `SMART_SCALE_CATALOG_YOLO`.
