from __future__ import annotations

import csv
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

from smart_scale.domain import ProductMatch

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
            matches.append(
                ProductMatch(
                    product_id=self.ids[idx],
                    name=str(meta.get("name") or meta.get("category") or self.ids[idx]),
                    score=float(scores[idx]),
                    price_per_gram=_safe_float(meta.get("price_per_gram") or meta.get("price")),
                    metadata=meta,
                )
            )
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
            matches.append(
                ProductMatch(
                    product_id=self.ids[idx],
                    name=str(meta.get("name") or meta.get("category") or self.ids[idx]),
                    score=float(score),
                    price_per_gram=_safe_float(meta.get("price_per_gram") or meta.get("price")),
                    metadata=meta,
                )
            )
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

    def __init__(self, dsn: str, table: str = "product_embeddings") -> None:
        if psycopg is None:
            raise RuntimeError("psycopg не установлен.")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError(f"Некорректное имя таблицы: {table}")

        self.dsn = dsn
        self.table = table

    def count(self) -> int:
        query = f"SELECT COUNT(*) FROM {self.table}"
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cursor:
            cursor.execute(query)
            row = cursor.fetchone()
        return int(row[0]) if row else 0

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[ProductMatch]:
        vector_literal = "[" + ",".join(f"{value:.8f}" for value in np.asarray(query_embedding, dtype=np.float32)) + "]"
        query = f"""
            SELECT
                product_id,
                COALESCE(name, metadata->>'category', product_id) AS product_name,
                price_per_gram,
                metadata,
                1 - (embedding <=> %s::vector) AS score
            FROM {self.table}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        with psycopg.connect(self.dsn) as conn, conn.cursor() as cursor:
            cursor.execute(query, (vector_literal, vector_literal, top_k))
            rows = cursor.fetchall()

        matches: list[ProductMatch] = []
        for product_id, name, price_per_gram, metadata, score in rows:
            matches.append(
                ProductMatch(
                    product_id=str(product_id),
                    name=str(name),
                    score=float(score),
                    price_per_gram=_safe_float(price_per_gram),
                    metadata=dict(metadata or {}),
                )
            )
        return matches


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
                metadata[Path(filename).name.lower()] = {
                    "name": row.get("name"),
                    "price_per_gram": _safe_float(row.get("price_per_gram") or row.get("price")),
                    "product_id": row.get("product_id"),
                    "category": row.get("category"),
                }
        return metadata

    def _build_metadata(
        self,
        image_path: Path,
        default_category: str,
        filename_to_meta: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        record = filename_to_meta.get(image_path.name.lower(), {})
        category = str(record.get("category") or default_category)
        name = str(record.get("name") or category)
        product_id = str(record.get("product_id") or f"{category}/{image_path.stem}")
        return {
            "product_id": product_id,
            "category": category,
            "name": name,
            "filename": image_path.name,
            "path": str(image_path),
            "price_per_gram": _safe_float(record.get("price_per_gram")),
        }


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
