# PgVector Bootstrap

Проект переведён на каталог `PostgreSQL + pgvector` со схемой:

- `product_id`
- `embedding`
- `product_type`
- `product_sort`
- `price_rub_per_kg`

## Что нужно для первичного заполнения

1. Поднять PostgreSQL с расширением `pgvector`.
2. Сохранить датасет PackEat локально.
3. Указать путь к датасету и строку подключения через переменные окружения.
4. Запустить bootstrap-команду.

## Быстрый старт через Docker

```powershell
docker compose -f docker-compose.pgvector.yml up -d
```

Строка подключения по умолчанию:

```text
postgresql://smart_scale:smart_scale@localhost:5433/smart_scale
```

## Переменные окружения

```powershell
$env:SMART_SCALE_VECTOR_BACKEND = "pgvector"
$env:SMART_SCALE_PGVECTOR_DSN = "postgresql://smart_scale:smart_scale@localhost:5433/smart_scale"
$env:SMART_SCALE_PGVECTOR_TABLE = "product_embeddings"
$env:SMART_SCALE_DATASET_DIR = "$PWD\varieties_classification_dataset"
$env:SMART_SCALE_PRICE_CATALOG = "$PWD\data\product_prices.py"
$env:SMART_SCALE_SAMPLES_PER_SORT = "5"
```

## Загрузка эталонного каталога

```powershell
smart-scale-bootstrap
```

Команда:

- создаёт таблицу с расширением `pgvector`, если их ещё нет;
- ищет изображения сортов в PackEat;
- берёт по `5` изображений на каждый сорт;
- строит эмбеддинги и загружает их в PostgreSQL.

Если в каком-то сорте изображений меньше пяти или сорт не найден в датасете, bootstrap завершится ошибкой.
