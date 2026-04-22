from __future__ import annotations

import csv
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

from smart_scale.domain import ProductMatch
from smart_scale.ml.catalog_seed import split_sort_label

try:
    import faiss
except ImportError:  # pragma: no cover - optional runtime deps
    faiss = None

try:
    import psycopg
except ImportError:  # pragma: no cover - optional runtime deps
    psycopg = None


class FileVectorStore:
    """Pure numpy KNN store persisted as a single snapshot file."""

    def __init__(self, dim: int, snapshot_path: str | Path) -> None:
        self.dim = dim
        self.snapshot_path = Path(snapshot_path)
        self.ids: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self.embeddings = np.empty((0, dim), dtype=np.float32)

    def count(self) -> int:
        return len(self.ids)

    def clear(self) -> None:
        self.ids.clear()
        self.metadata.clear()
        self.embeddings = np.empty((0, self.dim), dtype=np.float32)

    def add_batch(self, ids: list[str], embeddings: np.ndarray, metadata_list: list[dict[str, Any]]) -> None:
        vectors = _normalize_rows(embeddings, self.dim)
        if len(ids) != len(vectors) or len(metadata_list) != len(vectors):
            raise ValueError("Количество ids, векторов и metadata должно совпадать.")

        if self.count() == 0:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack([self.embeddings, vectors])
        self.ids.extend(ids)
        self.metadata.extend(metadata_list)

    def replace_catalog(self, ids: list[str], embeddings: np.ndarray, metadata_list: list[dict[str, Any]]) -> None:
        self.clear()
        if ids:
            self.add_batch(ids, embeddings, metadata_list)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[ProductMatch]:
        if self.count() == 0:
            return []

        query = _normalize_rows(query_embedding, self.dim)[0]
        scores = self.embeddings.dot(query)
        limit = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:limit]

        matches: list[ProductMatch] = []
        for idx in top_indices:
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            matches.append(_build_product_match(self.ids[idx], meta, float(scores[idx])))
        return matches

    def save(self, snapshot_path: str | Path | None = None) -> None:
        path = Path(snapshot_path or self.snapshot_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "dim": self.dim,
                    "ids": self.ids,
                    "metadata": self.metadata,
                    "embeddings": self.embeddings,
                },
                handle,
            )

    def load(self, snapshot_path: str | Path | None = None) -> None:
        path = Path(snapshot_path or self.snapshot_path)
        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("rb") as handle:
            data = pickle.load(handle)

        dim = int(data.get("dim", self.dim))
        embeddings = np.asarray(data.get("embeddings", np.empty((0, dim), dtype=np.float32)), dtype=np.float32)
        if embeddings.size == 0:
            embeddings = np.empty((0, self.dim), dtype=np.float32)
        elif embeddings.shape[1] != self.dim:
            raise ValueError(f"Несовпадение размерности снапшота: ожидалось {self.dim}, получено {embeddings.shape[1]}")

        self.ids = list(data.get("ids", []))
        self.metadata = list(data.get("metadata", []))
        self.embeddings = embeddings


class FaissVectorStore:
    def __init__(self, dim: int, snapshot_path: str | Path) -> None:
        self.dim = dim
        self.snapshot_path = Path(snapshot_path)
        self.ids: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self.index = faiss.IndexFlatIP(dim) if faiss is not None else None

    def count(self) -> int:
        if self.index is None:
            return len(self.ids)
        return int(getattr(self.index, "ntotal", len(self.ids)))

    def clear(self) -> None:
        self.ids.clear()
        self.metadata.clear()
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.dim)

    def add_batch(self, ids: list[str], embeddings: np.ndarray, metadata_list: list[dict[str, Any]]) -> None:
        if faiss is None or self.index is None:
            raise RuntimeError("faiss не установлен.")

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Несовпадение размерности: ожидалось {self.dim}, получено {vectors.shape[1]}")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = vectors / norms

        self.index.add(normalized)
        self.ids.extend(ids)
        self.metadata.extend(metadata_list)

    def replace_catalog(self, ids: list[str], embeddings: np.ndarray, metadata_list: list[dict[str, Any]]) -> None:
        self.clear()
        if ids:
            self.add_batch(ids, embeddings, metadata_list)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[ProductMatch]:
        if faiss is None or self.index is None or self.count() == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError(f"Несовпадение размерности запроса: ожидалось {self.dim}, получено {query.shape[1]}")

        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        query = query / norm

        scores, indices = self.index.search(query, top_k)
        matches: list[ProductMatch] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            matches.append(_build_product_match(self.ids[idx], meta, float(score)))
        return matches

    def save(self, snapshot_path: str | Path | None = None) -> None:
        if faiss is None or self.index is None:
            raise RuntimeError("faiss не установлен.")

        path = Path(snapshot_path or self.snapshot_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path) + ".index")
        with path.open("wb") as handle:
            pickle.dump({"ids": self.ids, "metadata": self.metadata}, handle)

    def load(self, snapshot_path: str | Path | None = None) -> None:
        if faiss is None:
            raise RuntimeError("faiss не установлен.")

        path = Path(snapshot_path or self.snapshot_path)
        index_path = Path(str(path) + ".index")
        if not path.exists() or not index_path.exists():
            raise FileNotFoundError(path)

        self.index = faiss.read_index(str(index_path))
        with path.open("rb") as handle:
            data = pickle.load(handle)

        self.ids = list(data.get("ids", []))
        self.metadata = list(data.get("metadata", []))


class PgVectorStore:
    """Search backend for PostgreSQL + pgvector."""

    REQUIRED_COLUMNS = {"product_id", "embedding", "product_type", "product_sort", "price_rub_per_kg"}

    def __init__(
        self,
        dsn: str,
        dim: int,
        table: str = "product_embeddings",
        recreate_on_dimension_mismatch: bool = False,
    ) -> None:
        if psycopg is None:
            raise RuntimeError("psycopg не установлен.")
        if dim <= 0:
            raise ValueError("dim должен быть больше нуля.")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError(f"Некорректное имя таблицы: {table}")

        self.dsn = dsn
        self.dim = dim
        self.table = table
        self.recreate_on_dimension_mismatch = recreate_on_dimension_mismatch
        self._schema_ready = False

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return

        with psycopg.connect(self.dsn) as conn, conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            if not self._table_exists(cursor):
                cursor.execute(self._create_table_sql())
                conn.commit()
                self._schema_ready = True
                return

            columns = self._fetch_columns(cursor)
            missing_identity = {"product_id", "embedding"} - set(columns)
            if missing_identity:
                raise RuntimeError(
                    "Существующая таблица pgvector не может быть мигрирована автоматически. "
                    f"Отсутствуют обязательные колонки: {', '.join(sorted(missing_identity))}."
                )

            current_dim = self._fetch_embedding_dimension(cursor)
            if current_dim is not None and current_dim != self.dim:
                if self.recreate_on_dimension_mismatch:
                    cursor.execute(f"DROP TABLE {self.table}")
                    cursor.execute(self._create_table_sql())
                    conn.commit()
                    self._schema_ready = True
                    return
                raise RuntimeError(
                    f"Existing pgvector table {self.table} uses vector({current_dim}), "
                    f"but the current embedding model produces vector({self.dim})."
                )

            for statement in self._migration_statements(columns):
                cursor.execute(statement)

            migrated_columns = self._fetch_columns(cursor)
            missing_required = self.REQUIRED_COLUMNS - set(migrated_columns)
            if missing_required:
                raise RuntimeError(
                    "После миграции схема pgvector осталась неполной. "
                    f"Отсутствуют колонки: {', '.join(sorted(missing_required))}."
                )

            cursor.execute(
                f"""
                SELECT COUNT(*)
                FROM {self.table}
                WHERE product_type IS NULL
                   OR product_sort IS NULL
                   OR price_rub_per_kg IS NULL
                """
            )
            invalid_rows = int(cursor.fetchone()[0] or 0)
            if invalid_rows > 0:
                raise RuntimeError(
                    "Не удалось безопасно мигрировать существующую таблицу pgvector: "
                    f"{invalid_rows} строк(и) остались без product_type/product_sort/price_rub_per_kg."
                )

            cursor.execute(f"ALTER TABLE {self.table} ALTER COLUMN product_type SET NOT NULL")
            cursor.execute(f"ALTER TABLE {self.table} ALTER COLUMN product_sort SET NOT NULL")
            cursor.execute(f"ALTER TABLE {self.table} ALTER COLUMN price_rub_per_kg SET NOT NULL")
            conn.commit()
        self._schema_ready = True

    def _create_table_sql(self) -> str:
        return f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                product_id TEXT PRIMARY KEY,
                embedding VECTOR({self.dim}) NOT NULL,
                product_type TEXT NOT NULL,
                product_sort TEXT NOT NULL,
                price_rub_per_kg DOUBLE PRECISION NOT NULL
            );
        """

    def _table_exists(self, cursor: Any) -> bool:
        cursor.execute("SELECT to_regclass(%s)", (self.table,))
        row = cursor.fetchone()
        return bool(row and row[0])

    def _fetch_columns(self, cursor: Any) -> set[str]:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %s
            """,
            (self.table,),
        )
        return {str(row[0]) for row in cursor.fetchall()}

    def _fetch_embedding_dimension(self, cursor: Any) -> int | None:
        cursor.execute(
            """
            SELECT format_type(attribute.atttypid, attribute.atttypmod)
            FROM pg_attribute AS attribute
            WHERE attribute.attrelid = to_regclass(%s)
              AND attribute.attname = 'embedding'
              AND NOT attribute.attisdropped
            """,
            (self.table,),
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            return None

        match = re.fullmatch(r"vector\((\d+)\)", str(row[0]))
        return int(match.group(1)) if match else None

    def _migration_statements(self, columns: set[str]) -> list[str]:
        statements: list[str] = []
        if "product_type" not in columns:
            statements.append(f"ALTER TABLE {self.table} ADD COLUMN product_type TEXT")
        if "product_sort" not in columns:
            statements.append(f"ALTER TABLE {self.table} ADD COLUMN product_sort TEXT")
        if "price_rub_per_kg" not in columns:
            statements.append(f"ALTER TABLE {self.table} ADD COLUMN price_rub_per_kg DOUBLE PRECISION")

        normalized_id = "split_part(split_part(product_id, ':', 1), '/', 1)"
        metadata_product_type = "NULLIF(metadata->>'product_type', '')" if "metadata" in columns else "NULL"
        metadata_product_sort = "NULLIF(metadata->>'product_sort', '')" if "metadata" in columns else "NULL"
        legacy_name = "NULLIF(name, '')" if "name" in columns else "NULL"
        legacy_price_per_gram = "price_per_gram * 1000.0" if "price_per_gram" in columns else "NULL"
        legacy_price = "price" if "price" in columns else "NULL"

        statements.append(
            f"""
            UPDATE {self.table}
            SET
                product_type = COALESCE(
                    NULLIF(product_type, ''),
                    {metadata_product_type},
                    split_part({normalized_id}, '_', 1),
                    {legacy_name},
                    {normalized_id},
                    product_id
                ),
                product_sort = COALESCE(
                    NULLIF(product_sort, ''),
                    {metadata_product_sort},
                    CASE
                        WHEN position('_' IN {normalized_id}) > 0
                        THEN substring({normalized_id} FROM position('_' IN {normalized_id}) + 1)
                        ELSE {normalized_id}
                    END,
                    {legacy_name},
                    {normalized_id},
                    product_id
                ),
                price_rub_per_kg = COALESCE(
                    price_rub_per_kg,
                    {legacy_price_per_gram},
                    {legacy_price}
                )
            """
        )
        return statements

    def count(self) -> int:
        self.ensure_schema()
        query = f"SELECT COUNT(*) FROM {self.table}"
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cursor:
            cursor.execute(query)
            row = cursor.fetchone()
        return int(row[0]) if row else 0

    def clear(self) -> None:
        self.ensure_schema()
        query = f"TRUNCATE TABLE {self.table}"
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cursor:
            cursor.execute(query)
            conn.commit()

    def add_batch(self, ids: list[str], embeddings: np.ndarray, metadata_list: list[dict[str, Any]]) -> None:
        self.ensure_schema()
        payload = self._prepare_payload(ids, embeddings, metadata_list)
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cursor:
            cursor.executemany(self._upsert_sql(self.table), payload)
            conn.commit()

    def replace_catalog(self, ids: list[str], embeddings: np.ndarray, metadata_list: list[dict[str, Any]]) -> None:
        self.ensure_schema()
        payload = self._prepare_payload(ids, embeddings, metadata_list)
        stage_table = f"{self.table}_staging"

        with psycopg.connect(self.dsn) as conn, conn.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TEMP TABLE {stage_table} (
                    product_id TEXT PRIMARY KEY,
                    embedding VECTOR({self.dim}) NOT NULL,
                    product_type TEXT NOT NULL,
                    product_sort TEXT NOT NULL,
                    price_rub_per_kg DOUBLE PRECISION NOT NULL
                ) ON COMMIT DROP
                """
            )
            if payload:
                cursor.executemany(self._upsert_sql(stage_table), payload)
            cursor.execute(f"LOCK TABLE {self.table} IN ACCESS EXCLUSIVE MODE")
            cursor.execute(f"TRUNCATE TABLE {self.table}")
            if payload:
                cursor.execute(
                    f"""
                    INSERT INTO {self.table} (
                        product_id,
                        embedding,
                        product_type,
                        product_sort,
                        price_rub_per_kg
                    )
                    SELECT
                        product_id,
                        embedding,
                        product_type,
                        product_sort,
                        price_rub_per_kg
                    FROM {stage_table}
                    """
                )
            conn.commit()

    def _upsert_sql(self, table_name: str) -> str:
        return f"""
            INSERT INTO {table_name} (
                product_id,
                embedding,
                product_type,
                product_sort,
                price_rub_per_kg
            )
            VALUES (%s, %s::vector, %s, %s, %s)
            ON CONFLICT (product_id) DO UPDATE
            SET
                embedding = EXCLUDED.embedding,
                product_type = EXCLUDED.product_type,
                product_sort = EXCLUDED.product_sort,
                price_rub_per_kg = EXCLUDED.price_rub_per_kg
        """

    def _prepare_payload(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        metadata_list: list[dict[str, Any]],
    ) -> list[tuple[str, str, str, str, float]]:
        vectors = _normalize_rows(embeddings, self.dim)
        if len(ids) != len(vectors) or len(metadata_list) != len(vectors):
            raise ValueError("Количество ids, векторов и metadata должно совпадать.")

        payload: list[tuple[str, str, str, str, float]] = []
        for product_id, vector, metadata in zip(ids, vectors, metadata_list, strict=False):
            product_type = _coerce_text(metadata.get("product_type"), fallback=_infer_type_from_id(product_id))
            product_sort = _coerce_text(metadata.get("product_sort"), fallback=_infer_sort_from_id(product_id))
            price_rub_per_kg = _safe_float(
                metadata.get("price_rub_per_kg") or metadata.get("price_per_kg") or metadata.get("price")
            )
            if price_rub_per_kg is None:
                raise ValueError(f"Для продукта {product_id} не указана цена.")

            payload.append(
                (
                    product_id,
                    _vector_literal(vector),
                    product_type,
                    product_sort,
                    price_rub_per_kg,
                )
            )
        return payload

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[ProductMatch]:
        self.ensure_schema()
        query_vector = _normalize_rows(query_embedding, self.dim)[0]
        vector_literal = _vector_literal(query_vector)
        query = f"""
            SELECT
                product_id,
                product_type,
                product_sort,
                price_rub_per_kg,
                1 - (embedding <=> %s::vector) AS score
            FROM {self.table}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        with psycopg.connect(self.dsn) as conn, conn.cursor() as cursor:
            cursor.execute(query, (vector_literal, vector_literal, top_k))
            rows = cursor.fetchall()

        matches: list[ProductMatch] = []
        for product_id, product_type, product_sort, price_rub_per_kg, score in rows:
            matches.append(
                ProductMatch(
                    product_id=str(product_id),
                    product_type=str(product_type),
                    product_sort=str(product_sort),
                    score=float(score),
                    price_rub_per_kg=_safe_float(price_rub_per_kg),
                    metadata={
                        "product_type": str(product_type),
                        "product_sort": str(product_sort),
                        "price_rub_per_kg": _safe_float(price_rub_per_kg),
                    },
                )
            )
        return matches

    def save(self, *_args, **_kwargs) -> None:
        self.ensure_schema()

    def load(self, *_args, **_kwargs) -> None:
        self.ensure_schema()


class CatalogIndexBuilder:
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, embedder: Any, store: Any) -> None:
        self.embedder = embedder
        self.store = store

    def build(
        self,
        images_dir: str | Path,
        products_csv: str | Path | None = None,
        batch_size: int = 16,
    ) -> int:
        image_root = Path(images_dir)
        if not image_root.exists():
            raise FileNotFoundError(image_root)

        filename_to_meta = self._load_catalog_metadata(products_csv or image_root / "product.csv")
        self.store.clear()

        image_paths: list[Path] = []
        ids: list[str] = []
        metadata_list: list[dict[str, Any]] = []

        directories = [child for child in image_root.iterdir() if child.is_dir()]
        if directories:
            for directory in sorted(directories):
                for image_path in sorted(directory.iterdir()):
                    if image_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                        continue
                    meta = self._build_metadata(image_path, directory.name, filename_to_meta)
                    image_paths.append(image_path)
                    ids.append(meta["product_id"])
                    metadata_list.append(meta)
        else:
            for image_path in sorted(image_root.iterdir()):
                if image_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    continue
                meta = self._build_metadata(image_path, image_path.stem, filename_to_meta)
                image_paths.append(image_path)
                ids.append(meta["product_id"])
                metadata_list.append(meta)

        for start in range(0, len(image_paths), batch_size):
            chunk_paths = image_paths[start:start + batch_size]
            chunk_ids = ids[start:start + batch_size]
            chunk_meta = metadata_list[start:start + batch_size]
            embeddings = self.embedder.embed_batch(chunk_paths, batch_size=batch_size)
            self.store.add_batch(chunk_ids, embeddings, chunk_meta)

        return len(image_paths)

    def _load_catalog_metadata(self, csv_path: str | Path | None) -> dict[str, dict[str, Any]]:
        if csv_path is None:
            return {}

        path = Path(csv_path)
        if not path.exists():
            return {}

        metadata: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                filename = (row.get("filename") or row.get("image") or row.get("id") or "").strip()
                if not filename:
                    continue

                legacy_label = row.get("category") or row.get("name") or Path(filename).stem
                product_type = row.get("product_type")
                product_sort = row.get("product_sort")
                if not product_type or not product_sort:
                    product_type, product_sort = split_sort_label(str(legacy_label))

                metadata[Path(filename).name.lower()] = {
                    "product_id": row.get("product_id"),
                    "product_type": product_type,
                    "product_sort": product_sort,
                    "price_rub_per_kg": _safe_float(
                        row.get("price_rub_per_kg") or row.get("price_per_gram") or row.get("price")
                    ),
                }
        return metadata

    def _build_metadata(
        self,
        image_path: Path,
        default_label: str,
        filename_to_meta: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        record = filename_to_meta.get(image_path.name.lower(), {})
        product_type = str(record.get("product_type") or split_sort_label(default_label)[0])
        product_sort = str(record.get("product_sort") or split_sort_label(default_label)[1])
        product_id = str(record.get("product_id") or f"{product_type}_{product_sort}/{image_path.stem}")
        return {
            "product_id": product_id,
            "product_type": product_type,
            "product_sort": product_sort,
            "filename": image_path.name,
            "path": str(image_path),
            "price_rub_per_kg": _safe_float(record.get("price_rub_per_kg")),
        }


def _build_product_match(product_id: str, meta: dict[str, Any], score: float) -> ProductMatch:
    product_type = _coerce_text(meta.get("product_type"), fallback=_infer_type_from_id(product_id))
    product_sort = _coerce_text(meta.get("product_sort"), fallback=_infer_sort_from_id(product_id))
    return ProductMatch(
        product_id=product_id,
        product_type=product_type,
        product_sort=product_sort,
        score=score,
        price_rub_per_kg=_safe_float(meta.get("price_rub_per_kg") or meta.get("price_per_gram") or meta.get("price")),
        metadata=dict(meta),
    )


def _infer_type_from_id(product_id: str) -> str:
    normalized = str(product_id).split(":", 1)[0].split("/", 1)[0]
    return split_sort_label(normalized)[0]


def _infer_sort_from_id(product_id: str) -> str:
    normalized = str(product_id).split(":", 1)[0].split("/", 1)[0]
    return split_sort_label(normalized)[1]


def _coerce_text(value: Any, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _vector_literal(values: np.ndarray) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in np.asarray(values, dtype=np.float32)) + "]"


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _normalize_rows(array: np.ndarray, dim: int) -> np.ndarray:
    vectors = np.asarray(array, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.shape[1] != dim:
        raise ValueError(f"Несовпадение размерности: ожидалось {dim}, получено {vectors.shape[1]}")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
