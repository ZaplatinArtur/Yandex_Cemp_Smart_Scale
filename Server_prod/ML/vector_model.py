"""
╔══════════════════════════════════════════════════════════════╗
║  Модуль поиска фруктов/овощей по фото                       ║
║  DINOv2 (дообученная) + Векторная БД (заглушка)             ║
╠══════════════════════════════════════════════════════════════╣
║  Использование:                                              ║
║    1. Положите fruit_embedder_best.pth рядом со скриптом     ║
║    2. python fruit_search.py --build ./images_folder         ║
║    3. python fruit_search.py --search photo.jpg              ║
║    4. python fruit_search.py --compare img1.jpg img2.jpg     ║
║    5. python fruit_search.py --server                        ║
╚══════════════════════════════════════════════════════════════╝

pip install transformers torch torchvision Pillow
"""

import os
import json
import time
import pickle
import argparse
import numpy as np
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from torchvision import transforms

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def _json_default(o):
    """Helper for json.dumps to handle numpy types and Paths."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Type {type(o)} not serializable")


# ================================================================
# 1. МОДЕЛЬ
# ================================================================
class FruitEmbedder(nn.Module):
    """Дообученная DINOv2 + projection head."""

    def __init__(self, backbone_name="facebook/dinov2-small",
                 backbone_dim=384, embedding_dim=256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        embedding = self.head(cls_token)
        return F.normalize(embedding, p=2, dim=1)


class EmbeddingModel:
    """Обёртка: загрузка модели + извлечение эмбеддингов."""

    def __init__(self, checkpoint_path : str, device: str = "auto", onnx_model_path: Optional[str] = None):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Загрузка модели ({self.device})...")

        # If user passed an ONNX file as --model, treat it as onnx_model_path
        is_onnx_path = bool(checkpoint_path and os.path.isfile(checkpoint_path) and checkpoint_path.lower().endswith('.onnx'))
        if is_onnx_path and not onnx_model_path:
            onnx_model_path = checkpoint_path

        checkpoint = {}

        # Load checkpoint only when not using ONNX as the primary source
        if not onnx_model_path:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            # Try standard torch.load, fall back to weights_only=False when supported
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
            except Exception as e:
                try:
                    # Some PyTorch versions changed default weights_only behavior
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                except TypeError:
                    raise RuntimeError(f"Failed to load checkpoint '{checkpoint_path}': {e}")
                except Exception as e2:
                    raise RuntimeError(f"Failed to load checkpoint '{checkpoint_path}': {e2}")

        # Derive processor/backbone defaults from checkpoint (or use defaults)
        backbone_name = checkpoint.get("backbone_name", "facebook/dinov2-small")
        embedding_dim = checkpoint.get("embedding_dim", 256)

        # Image processor + simple preprocessing (same as before)
        self.processor = AutoImageProcessor.from_pretrained(backbone_name)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])

        self.classes = checkpoint.get("classes", [])
        self.embedding_dim = embedding_dim

        # ONNX runtime support (optional). If provided, prefer ONNX for inference.
        self.onnx_session = None
        self.use_onnx = False
        self.onnx_input_name = None
        self.onnx_output_name = None

        if onnx_model_path:
            try:
                import onnxruntime as ort
                available = ort.get_available_providers()
                providers = []
                if self.device.type == 'cuda' and 'CUDAExecutionProvider' in available:
                    providers.append('CUDAExecutionProvider')
                if 'CPUExecutionProvider' in available:
                    providers.append('CPUExecutionProvider')
                providers = providers or None
                self.onnx_session = ort.InferenceSession(onnx_model_path, providers=providers)
                self.onnx_input_name = self.onnx_session.get_inputs()[0].name
                self.onnx_output_name = self.onnx_session.get_outputs()[0].name

                # Try a quick dummy inference to infer output dim if possible
                try:
                    dummy_img = Image.new('RGB', (224, 224))
                    proc = self.processor(images=self.preprocess(dummy_img), return_tensors='np')
                    dummy = proc['pixel_values'].astype(np.float32)
                    if dummy.ndim == 3:
                        dummy = np.expand_dims(dummy, 0)
                    res = self.onnx_session.run(None, {self.onnx_input_name: dummy})
                    if res and hasattr(res[0], 'shape'):
                        out_shape = res[0].shape
                        if len(out_shape) >= 2:
                            self.embedding_dim = int(out_shape[1])
                except Exception:
                    pass

                self.use_onnx = True
                self.model = None
                print(f"  ONNX model loaded: {onnx_model_path} (embedding_dim={self.embedding_dim})")
            except Exception as e:
                print(f"  Warning: failed to load ONNX model '{onnx_model_path}': {e}. Falling back to PyTorch model.")
                self.onnx_session = None

        if not self.use_onnx:
            # Build PyTorch model as before
            self.model = FruitEmbedder(
                backbone_name=backbone_name,
                embedding_dim=self.embedding_dim,
            )

            # Try various common checkpoint layouts
            if isinstance(checkpoint, dict):
                # Full model/state dict keys
                if 'state_dict' in checkpoint:
                    try:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    except Exception:
                        try:
                            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                        except Exception:
                            pass

                if 'model_state_dict' in checkpoint:
                    try:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    except Exception:
                        try:
                            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        except Exception:
                            pass

                head_state = checkpoint.get('head_state_dict')
                if head_state is not None:
                    try:
                        self.model.head.load_state_dict(head_state)
                    except Exception:
                        try:
                            self.model.head.load_state_dict(head_state, strict=False)
                        except Exception:
                            pass

                if 'backbone_state_dict' in checkpoint:
                    try:
                        self.model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
                    except Exception:
                        pass

                # Raw state_dict (fallback)
                if any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    try:
                        self.model.load_state_dict(checkpoint)
                    except Exception:
                        try:
                            self.model.load_state_dict(checkpoint, strict=False)
                        except Exception:
                            pass

            self.model.to(self.device)
            self.model.eval()
            print(f"  PyTorch model ready: dim={self.embedding_dim}, классов={len(self.classes)}")

    def get_embedding(self, image) -> np.ndarray:
        """
        Извлечь эмбеддинг из изображения.
        image: str (путь) | PIL.Image | np.ndarray
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        image = self.preprocess(image)

        # ONNX path
        if self.use_onnx and self.onnx_session is not None:
            inputs = self.processor(images=image, return_tensors="np")
            pixel_values = inputs["pixel_values"].astype(np.float32)
            if pixel_values.ndim == 3:
                pixel_values = np.expand_dims(pixel_values, 0)
            pixel_values = np.ascontiguousarray(pixel_values)
            try:
                ort_inputs = {self.onnx_input_name: pixel_values}
                outputs = self.onnx_session.run(None, ort_inputs)
                embedding = outputs[0]
            except Exception as e:
                raise RuntimeError(f"ONNX inference failed: {e}")

            # ensure L2-normalized
            if embedding.ndim == 2:
                norms = np.linalg.norm(embedding, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embedding = embedding / norms
            return embedding.flatten()

        # PyTorch path (fallback)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            embedding = self.model(pixel_values)

        return embedding.cpu().numpy().flatten()

    def get_embeddings_batch(self, images: list, batch_size: int = 32) -> np.ndarray:
        """Пакетное извлечение эмбеддингов."""
        all_embeddings = []

        # If ONNX available, run batches via ONNX Runtime
        if self.use_onnx and self.onnx_session is not None:
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                processed = []
                for img in batch_images:
                    if isinstance(img, str):
                        img = Image.open(img).convert("RGB")
                    img = self.preprocess(img)
                    inputs = self.processor(images=img, return_tensors="np")
                    processed.append(inputs["pixel_values"].squeeze(0))

                pixel_values = np.stack(processed).astype(np.float32)
                pixel_values = np.ascontiguousarray(pixel_values)
                try:
                    outputs = self.onnx_session.run(None, {self.onnx_input_name: pixel_values})
                    embeddings = outputs[0]
                except Exception as e:
                    raise RuntimeError(f"ONNX batch inference failed: {e}")

                # normalize
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embeddings = embeddings / norms
                all_embeddings.append(embeddings)

            if not all_embeddings:
                return np.empty((0, self.embedding_dim), dtype=np.float32)

            return np.vstack(all_embeddings)

        # PyTorch fallback
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            processed = []

            for img in batch_images:
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                img = self.preprocess(img)
                inputs = self.processor(images=img, return_tensors="pt")
                processed.append(inputs["pixel_values"].squeeze(0))

            pixel_values = torch.stack(processed).to(self.device)

            with torch.no_grad():
                embeddings = self.model(pixel_values)

            all_embeddings.append(embeddings.cpu().numpy())

        if not all_embeddings:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        return np.vstack(all_embeddings)

    def compare(self, image1, image2) -> float:
        """Cosine similarity между двумя изображениями. (диапазон [-1..1])"""
        e1 = self.get_embedding(image1)
        e2 = self.get_embedding(image2)
        return float(np.dot(e1, e2))


# ================================================================
# 2. ВЕКТОРНАЯ БАЗА ДАННЫХ (заглушка)
# ================================================================
class VectorDB:
    """
    Заглушка для векторной БД.

    Хранит эмбеддинги в numpy-массиве, ищет через cosine similarity.
    Легко заменяется на настоящую БД:
      - Qdrant     → class QdrantDB(VectorDB)
      - ChromaDB   → class ChromaDB(VectorDB)
      - Pinecone   → class PineconeDB(VectorDB)
      - FAISS      → class FaissDB(VectorDB)
      - Milvus     → class MilvusDB(VectorDB)
      - Weaviate   → class WeaviateDB(VectorDB)

    Интерфейс (эти методы нужно реализовать в настоящей БД):
      - add(id, embedding, metadata)
      - search(query_embedding, top_k) -> results
      - delete(id)
      - save(path) / load(path)
    """

    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict] = []
        self.ids: List[str] = []
        print("  VectorDB: заглушка (numpy cosine similarity)")

    def add(self, item_id: str, embedding: np.ndarray, metadata: Dict = None):
        """Добавить элемент в базу."""
        # Проверка на дубликат
        if item_id in self.ids:
            idx = self.ids.index(item_id)
            self.embeddings[idx] = embedding
            self.metadata[idx] = metadata or {}
            return

        self.ids.append(item_id)
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})

    def add_batch(self, ids: List[str], embeddings: np.ndarray,
                  metadata_list: List[Dict] = None):
        """Пакетное добавление."""
        if metadata_list is None:
            metadata_list = [{}] * len(ids)

        for i, (item_id, emb, meta) in enumerate(zip(ids, embeddings, metadata_list)):
            self.add(item_id, emb, meta)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Поиск ближайших соседей.
        Возвращает: [{"id": ..., "score": ..., "metadata": ...}, ...]
        """
        if len(self.embeddings) == 0:
            return []

        # Stack/convert to numeric array safely
        db_matrix = np.asarray(self.embeddings, dtype=np.float32)
        query_vec = np.asarray(query_embedding, dtype=np.float32)

        if query_vec.ndim == 2 and query_vec.shape[0] == 1:
            query_vec = query_vec.flatten()

        if db_matrix.ndim != 2 or query_vec.ndim != 1:
            raise ValueError(f"Bad embeddings shape: db={db_matrix.shape}, query={query_vec.shape}")

        if db_matrix.shape[1] != query_vec.shape[0]:
            raise ValueError(f"Embedding dimension mismatch: db_dim={db_matrix.shape[1]}, query_dim={query_vec.shape[0]}")

        # Cosine similarity (вектора уже L2-нормализованы)
        scores = db_matrix.dot(query_vec)

        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "id": self.ids[idx],
                "score": float(scores[idx]),
                "metadata": self.metadata[idx],
            })

        return results

    def delete(self, item_id: str) -> bool:
        """Удалить элемент по ID."""
        if item_id not in self.ids:
            return False

        idx = self.ids.index(item_id)
        self.ids.pop(idx)
        self.embeddings.pop(idx)
        self.metadata.pop(idx)
        return True

    def count(self) -> int:
        return len(self.ids)

    def save(self, path: str):
        """Сохранить базу на диск."""
        dirn = os.path.dirname(path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)

        data = {
            "ids": self.ids,
            "embeddings": np.array(self.embeddings) if self.embeddings else np.array([]),
            "metadata": self.metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  База сохранена: {path} ({size_mb:.1f} MB, {self.count()} элементов)")

    def load(self, path: str):
        """Загрузить базу с диска."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.ids = data["ids"]
        self.embeddings = list(data["embeddings"]) if len(data["embeddings"]) > 0 else []
        self.metadata = data["metadata"]
        print(f"  База загружена: {path} ({self.count()} элементов)")

    def clear(self):
        """Очистить базу."""
        self.embeddings.clear()
        self.metadata.clear()
        self.ids.clear()

    def stats(self) -> Dict:
        """Статистика по базе."""
        categories = {}
        for meta in self.metadata:
            cat = meta.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_items": self.count(),
            "embedding_dim": self.embeddings[0].shape[0] if self.embeddings else 0,
            "categories": categories,
            "top_categories": sorted(categories.items(), key=lambda x: -x[1])[:10],
        }


# ================================================================
# 3. ПОИСКОВЫЙ ДВИЖОК
# ================================================================
class FruitSearchEngine:
    """
    Объединяет модель + векторную БД в удобный интерфейс.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, checkpoint_path: str, db_path: str = None, device: str = "auto",
                 db_backend: str = "faiss", db_config: Optional[Dict] = None):
        self.model = EmbeddingModel(checkpoint_path, device)

        # Choose DB backend: 'numpy' (VectorDB), 'faiss', or 'qdrant'
        # db_backend: optional string; db_config: optional dict for backend-specific params
        def _init_db(db_backend: str = "faiss", db_config: Optional[Dict] = None):
            # Enforce FaissDB as the only supported backend for production prototype.
            backend = (db_backend or "faiss").lower()
            if backend != "faiss":
                print(f"  Warning: requested db_backend='{db_backend}' ignored. Only 'faiss' is supported.")

            try:
                # Import FaissDB and ensure underlying faiss library is available
                from db_adapters import FaissDB, faiss as _faiss
            except Exception as e:
                raise RuntimeError(f"FaissDB adapter not available: {e}") from e

            if _faiss is None:
                raise ImportError("Faiss library is not installed. Install 'faiss-cpu' or 'faiss-gpu' to use FaissDB.")

            print(f"  Инициализация FaissDB (dim={self.model.embedding_dim})")
            return FaissDB(dim=self.model.embedding_dim)

        # default backend is 'numpy' — may be overridden by caller
        self.db = _init_db(db_backend, db_config)

        if db_path and os.path.exists(db_path):
            try:
                self.db.load(db_path)
            except Exception as e:
                print(f"  Warning: failed to load DB from {db_path}: {e}")

    def build_database(self, images_dir: str, batch_size: int = 32, products_csv: Optional[str] = None):
        """
        Построить БД из папки с изображениями.

        Поддерживает структуру:
          images_dir/
            category1/
              img1.jpg
            category2/
              img2.jpg
          ИЛИ плоскую:
          images_dir/
            img1.jpg
            img2.jpg
        """
        print(f"\nПостроение базы из: {images_dir}")
        start_time = time.time()

        # Попытка загрузить соответствие filename -> product_id из CSV.
        filename_to_pid = {}
        category_to_pid = {}
        csv_path = None
        if products_csv:
            csv_path = products_csv
        else:
            # accept both 'products.csv' and 'product.csv' (case-sensitive filename)
            candidates = [os.path.join(images_dir, "products.csv"), os.path.join(images_dir, "product.csv")]
            for c in candidates:
                if os.path.exists(c):
                    csv_path = c
                    break

        if csv_path:
            try:
                with open(csv_path, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        pid = row.get("product_id") or row.get("product") or row.get("productId")
                        fname = row.get("filename") or row.get("id") or row.get("image")
                        category = row.get("category")
                        if fname and pid:
                            # normalize filename: strip quotes/spaces and index by both basename and original (lowercased)
                            fname_norm = str(fname).strip().strip('"').strip("'")
                            bname = Path(fname_norm).name
                            filename_to_pid[bname.lower()] = pid
                            filename_to_pid[fname_norm.lower()] = pid
                        elif category and pid:
                            category_to_pid[category.lower()] = pid
                print(f"  Загружено отображение product_id из: {csv_path} ({len(filename_to_pid)} файлов, {len(category_to_pid)} категорий)")
            except Exception as e:
                print(f"  Warning: failed to parse products CSV {csv_path}: {e}")

        image_paths = []
        image_ids = []
        image_metadata = []

        root = Path(images_dir)

        if not root.exists() or not root.is_dir():
            print(f"  Путь не найден или не директория: {images_dir}")
            return

        # Проверяем: вложенные папки (категории) или плоская структура
        subdirs = [d for d in root.iterdir() if d.is_dir()]

        if subdirs:
            # Структура с категориями
            for category_dir in sorted(subdirs):
                category = category_dir.name
                for img_path in sorted(category_dir.iterdir()):
                    if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                        image_paths.append(str(img_path))
                        image_ids.append(f"{category}/{img_path.name}")
                        # определяем product_id: сначала по filename->pid из CSV (с нормализацией),
                        # затем по абсолютному пути из CSV, затем по категории, затем по префиксу filename (числа_)
                        bname = img_path.name
                        pid = filename_to_pid.get(bname.lower()) or filename_to_pid.get(str(img_path).lower())
                        if not pid:
                            pid = category_to_pid.get(category.lower()) or category_to_pid.get(category)
                        if not pid:
                            # пример: "123_apple.jpg" -> product_id = "123"
                            parts = img_path.name.split("_", 1)
                            if parts and parts[0].isdigit():
                                pid = parts[0]

                        image_metadata.append({
                            "category": category,
                            "filename": img_path.name,
                            "path": str(img_path),
                            "product_id": pid,
                        })
        else:
            # Плоская структура
            for img_path in sorted(root.iterdir()):
                if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    image_paths.append(str(img_path))
                    image_ids.append(img_path.name)
                    bname = img_path.name
                    pid = filename_to_pid.get(bname.lower()) or filename_to_pid.get(str(img_path).lower())
                    if not pid:
                        # попробовать извлечь из имени файла префикс
                        parts = img_path.name.split("_", 1)
                        if parts and parts[0].isdigit():
                            pid = parts[0]

                    image_metadata.append({
                        "category": "unknown",
                        "filename": img_path.name,
                        "path": str(img_path),
                        "product_id": pid,
                    })

        if not image_paths:
            print("  Изображения не найдены!")
            return

        print(f"  Найдено изображений: {len(image_paths)}")

        # Пакетное извлечение эмбеддингов
        processed = 0
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_ids = image_ids[i:i + batch_size]
            batch_meta = image_metadata[i:i + batch_size]

            try:
                embeddings = self.model.get_embeddings_batch(batch_paths, batch_size)
                self.db.add_batch(batch_ids, embeddings, batch_meta)
                processed += len(batch_paths)

                if processed % 200 == 0 or processed == len(image_paths):
                    elapsed = time.time() - start_time
                    speed = processed / elapsed
                    print(f"  Обработано: {processed}/{len(image_paths)} "
                          f"({speed:.0f} img/s)")
            except Exception as e:
                print(f"  Ошибка в батче {i}: {e}")
                # Обрабатываем по одному
                for path, item_id, meta in zip(batch_paths, batch_ids, batch_meta):
                    try:
                        emb = self.model.get_embedding(path)
                        self.db.add(item_id, emb, meta)
                        processed += 1
                    except Exception as e2:
                        print(f"    Пропущен {path}: {e2}")

        elapsed = time.time() - start_time
        print(f"\n  Готово! {self.db.count()} элементов за {elapsed:.1f}с")

    def search(self, query_image, top_k: int = 5) -> List[Dict]:
        """
        Найти похожие изображения.
        query_image: str (путь) | PIL.Image
        Возвращает список результатов с полями:
          - id, score, metadata (как раньше)
          - db_index: индекс записи в `self.db.ids` (если доступно)
          - name: категория или product_id или id (читаемое имя)
        """
        query_emb = self.model.get_embedding(query_image)
        results = self.db.search(query_emb, top_k)

        # attach DB index and human-readable name when available
        db_ids = getattr(self.db, "ids", None)
        out = []
        for r in results:
            item = dict(r)  # shallow copy
            item_meta = item.get("metadata", {}) or {}
            if db_ids and item.get("id") in db_ids:
                try:
                    item["db_index"] = db_ids.index(item.get("id"))
                except ValueError:
                    item["db_index"] = None
            else:
                item["db_index"] = None
            item["name"] = item_meta.get("category") or item_meta.get("product_id") or item.get("id")
            out.append(item)
        return out

    def compare(self, image1, image2) -> Dict:
        """Сравнить два изображения."""
        score = self.model.compare(image1, image2)
        return {
            "score": score,
            "percentage": f"{score * 100:.1f}%",
            "verdict": self._score_to_verdict(score),
        }

    def identify(self, image, top_k: int = 3) -> List[Dict]:
        """
        Определить что на фото по базе.
        Группирует результаты по категориям.
        """
        results = self.search(image, top_k=top_k * 3)

        # Группируем по категории, берём лучший score
        category_scores = {}
        for r in results:
            cat = r["metadata"].get("category", "unknown")
            if cat not in category_scores or r["score"] > category_scores[cat]["score"]:
                category_scores[cat] = r

        # Сортируем по score
        sorted_cats = sorted(category_scores.values(), key=lambda x: -x["score"])
        return sorted_cats[:top_k]

    def save_database(self, path: str):
        """Сохранить БД на диск."""
        dirn = os.path.dirname(path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        self.db.save(path)

    def load_database(self, path: str):
        """Загрузить БД с диска."""
        self.db.load(path)

    def stats(self) -> Dict:
        """Статистика базы."""
        return self.db.stats()

    @staticmethod
    def _score_to_verdict(score: float) -> str:
        if score > 0.90:
            return "Практически идентичны"
        elif score > 0.75:
            return "Очень похожи"
        elif score > 0.55:
            return "Похожи"
        elif score > 0.35:
            return "Слабое сходство"
        else:
            return "Не похожи"


# ================================================================
# 4. ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# ================================================================
def demo_basic():
    """Базовый пример — сравнение двух фото."""
    print("\n" + "=" * 60)
    print("  ДЕМО: Сравнение двух фото")
    print("=" * 60)

    engine = FruitSearchEngine("fruit_embedder_best.pth")
    result = engine.compare("apple.jpg", "banana.jpg")
    print(f"\n  Сходство: {result['percentage']}")
    print(f"  Вердикт:  {result['verdict']}")


def demo_build_and_search():
    """Построить базу из папки и искать."""
    print("\n" + "=" * 60)
    print("  ДЕМО: Построение базы + поиск")
    print("=" * 60)

    engine = FruitSearchEngine("fruit_embedder_best.pth")

    # 1. Строим базу
    engine.build_database("./fruits_images/")  # ваша папка с фото
    engine.save_database("fruits.db")

    # 2. Статистика
    stats = engine.stats()
    print(f"\n  Элементов в базе: {stats['total_items']}")
    print(f"  Категории: {stats['top_categories']}")

    # 3. Поиск
    results = engine.search("query_photo.jpg", top_k=5)
    print(f"\n  Результаты поиска:")
    for i, r in enumerate(results, 1):
        print(f"    {i}. [{r['score']:.3f}] {r['id']} "
              f"({r['metadata'].get('category', '?')})")


def demo_identify():
    """Определить что на фото."""
    print("\n" + "=" * 60)
    print("  ДЕМО: Идентификация продукта")
    print("=" * 60)

    engine = FruitSearchEngine("fruit_embedder_best.pth")
    engine.load_database("fruits.db")

    predictions = engine.identify("unknown_fruit.jpg", top_k=3)
    print(f"\n  Это скорее всего:")
    for i, p in enumerate(predictions, 1):
        cat = p["metadata"].get("category", "?")
        print(f"    {i}. {cat} (уверенность: {p['score']:.3f})")


def demo_api_server():
    """
    Простой HTTP-сервер для поиска.
    Запросы:
      POST /search   — поиск по фото
      POST /compare  — сравнение двух фото
      GET  /stats    — статистика
    """
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        import base64
        import io
    except ImportError:
        print("Для сервера нужен Python 3.7+")
        return

    engine = FruitSearchEngine("fruit_embedder_best.pth")

    db_path = "fruits.db"
    if os.path.exists(db_path):
        engine.load_database(db_path)

    class SearchHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            if self.path == "/search":
                data = json.loads(body)
                image_b64 = data["image"]
                top_k = data.get("top_k", 5)

                image_bytes = base64.b64decode(image_b64)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                results = engine.search(image, top_k=top_k)
                self._respond(200, {"results": results})

            elif self.path == "/compare":
                data = json.loads(body)
                img1_bytes = base64.b64decode(data["image1"])
                img2_bytes = base64.b64decode(data["image2"])

                img1 = Image.open(io.BytesIO(img1_bytes)).convert("RGB")
                img2 = Image.open(io.BytesIO(img2_bytes)).convert("RGB")

                result = engine.compare(img1, img2)
                self._respond(200, result)
            else:
                self._respond(404, {"error": "Not found"})

        def do_GET(self):
            if self.path == "/stats":
                self._respond(200, engine.stats())
            elif self.path == "/health":
                self._respond(200, {"status": "ok", "items": engine.db.count()})
            else:
                self._respond(404, {"error": "Not found"})

        def _respond(self, code, data):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                payload = json.dumps(data, ensure_ascii=False, default=_json_default)
            except TypeError:
                # Fallback: attempt to convert loosely
                payload = json.dumps(data, ensure_ascii=False, default=lambda o: str(o))
            self.wfile.write(payload.encode())

        def log_message(self, format, *args):
            print(f"  [{self.address_string()}] {args[0]}")

    port = 8080
    server = HTTPServer(("0.0.0.0", port), SearchHandler)
    print(f"\n  Сервер запущен: http://localhost:{port}")
    print(f"  Эндпоинты:")
    print(f"    POST /search   — поиск (body: {{image: base64, top_k: 5}})")
    print(f"    POST /compare  — сравнение (body: {{image1: b64, image2: b64}})")
    print(f"    GET  /stats    — статистика базы")
    print(f"    GET  /health   — проверка работы")
    print(f"\n  Ctrl+C для остановки\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Сервер остановлен")
        server.server_close()


# ================================================================
# 5. CLI ИНТЕРФЕЙС
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Поиск фруктов/овощей по фото",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python fruit_search.py --build ./Training/
  python fruit_search.py --search apple.jpg
  python fruit_search.py --search apple.jpg --top 10
  python fruit_search.py --compare apple.jpg banana.jpg
  python fruit_search.py --identify mystery.jpg
  python fruit_search.py --stats
  python fruit_search.py --server
        """,
    )

    parser.add_argument("--model", default="C:\\Users\\artur\\Desktop\\Yandex_Cemp\\fruit_embedder_final.pth",
                        help="Путь к чекпоинту модели")
    parser.add_argument("--db", default=os.path.join("Server_prod","DB","fruits.db"),
                        help="Путь к файлу базы данных")
    parser.add_argument("--db-backend", choices=["faiss"], default="faiss",
                        help="Выбрать бэкенд для векторной БД (только: faiss)")
    parser.add_argument("--build", type=str,
                        help="Построить базу из папки с изображениями")
    parser.add_argument("--search", type=str,
                        help="Найти похожие изображения")
    parser.add_argument("--compare", nargs=2, metavar=("IMG1", "IMG2"),
                        help="Сравнить два изображения")
    parser.add_argument("--identify", type=str,
                        help="Определить что на фото")
    parser.add_argument("--stats", action="store_true",
                        help="Показать статистику базы")
    parser.add_argument("--server", action="store_true",
                        help="Запустить HTTP-сервер")
    parser.add_argument("--top", type=int, default=5,
                        help="Количество результатов (default: 5)")

    args = parser.parse_args()

    if not any([args.build, args.search, args.compare,
                args.identify, args.stats, args.server]):
        parser.print_help()
        return

    # Resolve DB path relative to project root to avoid creating nested
    # Server_prod/ML/Server_prod when running from inside Server_prod/ML.
    db_path_arg = args.db
    if not os.path.isabs(db_path_arg):
        # project root: two levels up from this file (.. / .. / this_file)
        project_root = Path(__file__).resolve().parents[2]
        db_path = str(project_root.joinpath(db_path_arg))
    else:
        db_path = db_path_arg

    print(f"Using DB file: {db_path}")
    engine = FruitSearchEngine(args.model, db_path, db_backend=args.db_backend)

    if args.build:
        engine.build_database(args.build)
        engine.save_database(db_path)

    if args.search:
        results = engine.search(args.search, top_k=args.top)
        print(f"\nТоп-{args.top} похожих:")
        for i, r in enumerate(results, 1):
            cat = r["metadata"].get("category", "")
            pid = r["metadata"].get("product_id", "")
            pid_str = f" product_id:{pid}" if pid else ""
            meta_str = json.dumps(r["metadata"], ensure_ascii=False, default=_json_default)
            print(f"  {i}. [{r['score']:.3f}] {cat} — {r['id']}{pid_str}")
            print(f"     metadata: {meta_str}")

    if args.compare:
        result = engine.compare(args.compare[0], args.compare[1])
        print(f"\nСходство: {result['percentage']}")
        print(f"Вердикт:  {result['verdict']}")

    if args.identify:
        predictions = engine.identify(args.identify, top_k=args.top)
        print(f"\nЭто скорее всего:")
        for i, p in enumerate(predictions, 1):
            cat = p["metadata"].get("category", "?")
            pid = p["metadata"].get("product_id", "")
            pid_str = f" product_id:{pid}" if pid else ""
            meta_str = json.dumps(p["metadata"], ensure_ascii=False, default=_json_default)
            print(f"  {i}. {cat} ({p['score']:.3f}){pid_str}")
            print(f"     metadata: {meta_str}")

    if args.stats:
        stats = engine.stats()
        print(f"\nСтатистика базы:")
        print(f"  Элементов: {stats['total_items']}")
        print(f"  Размерность: {stats['embedding_dim']}")
        print(f"  Категории ({len(stats['categories'])}):")
        for cat, count in stats["top_categories"]:
            print(f"    {cat}: {count}")

    if args.server:
        demo_api_server()


# ================================================================
# 6. QUICK START (если запускаете в Jupyter / интерактивно)
# ================================================================
"""
# --- В Jupyter / Python скрипте: ---

from fruit_search import FruitSearchEngine

# Создать движок
engine = FruitSearchEngine("fruit_embedder_best.pth")

# Построить базу из папки с фото
engine.build_database("./Training/")
engine.save_database("fruits.db")

# Поиск
results = engine.search("my_photo.jpg", top_k=5)
for r in results:
    print(f"{r['score']:.3f}  {r['metadata']['category']}  {r['id']}")

# Сравнение
result = engine.compare("apple1.jpg", "apple2.jpg")
print(result)
# {'score': 0.92, 'percentage': '92.0%', 'verdict': 'Практически идентичны'}

# Идентификация
predictions = engine.identify("unknown.jpg", top_k=3)
for p in predictions:
    print(f"{p['metadata']['category']}: {p['score']:.3f}")

# Для следующего раза — просто загрузить базу
engine = FruitSearchEngine("fruit_embedder_best.pth", db_path="fruits.db")
results = engine.search("photo.jpg")


# --- Замена заглушки на настоящую БД: ---
#
# 1. Наследуйте от VectorDB
# 2. Реализуйте: add(), search(), delete(), save(), load()
# 3. Передайте в FruitSearchEngine:
#
#    engine = FruitSearchEngine("fruit_embedder_best.pth")
#    engine.db = QdrantDB(host="localhost", port=6333)
#    engine.build_database("./images/")
"""


if __name__ == "__main__":
    main()