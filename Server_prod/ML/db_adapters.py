"""
DB adapters for vector search: FaissDB and QdrantDB

This module provides minimal adapter classes that mirror the `VectorDB`
interface from `vector_model.py`. Imports of optional dependencies are
lazy/guarded so the module can be imported even if `faiss` or
`qdrant-client` are not installed. Instantiate the adapters only when the
corresponding libraries or services are available.

Usage examples:
    from db_adapters import FaissDB
    db = FaissDB(dim=256)
    db.add('id1', emb_vector, {'category': 'apple'})
    results = db.search(query_vector, top_k=5)

Note: Faiss adapter uses inner-product (`IndexFlatIP`) so embeddings
should be L2-normalized (the adapter normalizes inputs automatically).
"""

import os
import pickle
import numpy as np
from typing import List, Dict

# Minimal VectorDB base class used by adapters. Kept intentionally simple
# to avoid importing the heavier `vector_model` module here and to make
# this module import-safe even when optional dependencies are missing.
class VectorDB:
    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict] = []
        self.ids: List[str] = []

    def add(self, *args, **kwargs):
        raise NotImplementedError

    def add_batch(self, *args, **kwargs):
        raise NotImplementedError

    def search(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        raise NotImplementedError

    def count(self) -> int:
        """Return number of stored items (default implementation)."""
        return len(self.ids)


# FAISS adapter (optional)
try:
    import faiss
except Exception:
    faiss = None


class FaissDB(VectorDB):
    """Faiss-backed DB.

    - Uses IndexFlatIP (inner product) so embeddings are expected to be
      L2-normalized. The adapter normalizes inputs automatically.
    - Keeps a parallel `ids` and `metadata` list to map index positions
      to items. Save/load writes the faiss index + pickle for ids/metadata.
    """

    def __init__(self, dim: int, index_path: str = None):
        super().__init__()
        self.dim = dim
        self.index_path = index_path
        self.index = None
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.dim)

    def _ensure_index(self):
        if faiss is None:
            raise ImportError("faiss not installed. Install 'faiss-cpu' or 'faiss-gpu'.")
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)

    def add(self, item_id: str, embedding: np.ndarray, metadata: Dict = None):
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(emb)
        if norm == 0:
            raise ValueError("Zero vector cannot be indexed")
        emb = emb / norm
        if emb.shape[1] != self.dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.dim}, got {emb.shape[1]}")
        self._ensure_index()
        self.index.add(emb)
        self.ids.append(item_id)
        self.embeddings.append(emb.flatten())
        self.metadata.append(metadata or {})

    def add_batch(self, ids: List[str], embeddings: np.ndarray, metadata_list: List[Dict] = None):
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        if arr.shape[1] != self.dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.dim}, got {arr.shape[1]}")
        self._ensure_index()
        self.index.add(arr)
        if metadata_list is None:
            metadata_list = [{}] * arr.shape[0]
        for i, id_ in enumerate(ids):
            self.ids.append(id_)
            self.embeddings.append(arr[i].copy())
            self.metadata.append(metadata_list[i] if i < len(metadata_list) else {})

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if faiss is None:
            raise ImportError("faiss not installed.")
        if self.index is None or getattr(self.index, 'ntotal', 0) == 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        qn = np.linalg.norm(q)
        if qn == 0:
            raise ValueError("Zero query vector")
        q = q / qn
        D, I = self.index.search(q, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            results.append({
                "id": self.ids[idx],
                "score": float(score),
                "metadata": self.metadata[idx],
            })
        return results

    def count(self) -> int:
        """Return number of indexed vectors."""
        if faiss is not None and self.index is not None:
            try:
                return int(getattr(self.index, 'ntotal', len(self.ids)))
            except Exception:
                return len(self.ids)
        return len(self.ids)

    def save(self, path: str):
        if faiss is None:
            raise ImportError("faiss not installed.")
        dirn = os.path.dirname(path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        idx_path = path + ".index"
        faiss.write_index(self.index, idx_path)
        data = {"ids": self.ids, "metadata": self.metadata}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        if faiss is None:
            raise ImportError("faiss not installed.")
        idx_path = path + ".index"
        if not os.path.exists(idx_path):
            raise FileNotFoundError(idx_path)
        self.index = faiss.read_index(idx_path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.ids = data.get("ids", [])
        self.metadata = data.get("metadata", [])
        # embeddings can be left empty (can reconstruct by querying index if needed)
        self.embeddings = []


# Qdrant adapter (optional)
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
except Exception:
    QdrantClient = None
    rest = None


class QdrantDB(VectorDB):
    """Minimal Qdrant adapter. Requires a running Qdrant instance and
    `qdrant-client` installed. The implementation uses `upsert` and
    `search` for basic operations.
    """

    def __init__(self, dim: int, collection_name: str = "fruits", host: str = "localhost", port: int = 6333, prefer_grpc: bool = False):
        super().__init__()
        self.dim = dim
        self.collection_name = collection_name
        if QdrantClient is None:
            self.client = None
        else:
            self.client = QdrantClient(host=host, port=port, prefer_grpc=prefer_grpc)
            # create collection if it doesn't exist
            try:
                self.client.get_collection(self.collection_name)
            except Exception:
                if rest is None:
                    raise RuntimeError("qdrant-client models unavailable")
                self.client.recreate_collection(self.collection_name, vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE))

    def add(self, item_id: str, embedding: np.ndarray, metadata: Dict = None):
        if self.client is None:
            raise ImportError("qdrant-client not installed.")
        vec = np.asarray(embedding, dtype=np.float32).tolist()
        payload = metadata or {}
        point = rest.PointStruct(id=item_id, vector=vec, payload=payload)
        self.client.upsert(collection_name=self.collection_name, points=[point])
        self.ids.append(item_id)
        self.metadata.append(payload)
        self.embeddings.append(np.asarray(vec, dtype=np.float32))

    def add_batch(self, ids: List[str], embeddings: np.ndarray, metadata_list: List[Dict] = None):
        if self.client is None:
            raise ImportError("qdrant-client not installed.")
        if metadata_list is None:
            metadata_list = [{}] * len(ids)
        points = []
        for id_, emb, meta in zip(ids, embeddings, metadata_list):
            points.append(rest.PointStruct(id=id_, vector=np.asarray(emb, dtype=np.float32).tolist(), payload=meta or {}))
        self.client.upsert(collection_name=self.collection_name, points=points)
        for id_, meta, emb in zip(ids, metadata_list, embeddings):
            self.ids.append(id_)
            self.metadata.append(meta or {})
            self.embeddings.append(np.asarray(emb, dtype=np.float32))

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.client is None:
            raise ImportError("qdrant-client not installed.")
        q = np.asarray(query_embedding, dtype=np.float32).tolist()
        hits = self.client.search(collection_name=self.collection_name, query_vector=q, limit=top_k)
        results = []
        for h in hits:
            # ScoredPoint: h.id, h.score, h.payload
            results.append({
                "id": str(h.id),
                "score": float(h.score),
                "metadata": getattr(h, 'payload', {}) or {},
            })
        return results

    def save(self, path: str):
        # Qdrant persists data server-side; save mapping locally if desired
        dirn = os.path.dirname(path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        data = {"ids": self.ids, "metadata": self.metadata}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.ids = data.get("ids", [])
        self.metadata = data.get("metadata", [])
        self.embeddings = []
