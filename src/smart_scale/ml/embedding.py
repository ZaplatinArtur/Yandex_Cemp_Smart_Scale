from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModel
except ImportError:  # pragma: no cover - optional runtime deps
    torch = None
    nn = None
    F = None
    AutoImageProcessor = None
    AutoModel = None


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class DinoV2BackboneEmbedder(nn.Module if nn is not None else object):
    def __init__(self, model_name: str = "facebook/dinov2-small") -> None:
        if nn is None or AutoModel is None or F is None:
            raise RuntimeError("PyTorch/Transformers dependencies are not installed.")

        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

    def forward(self, pixel_values: Any) -> Any:
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return F.normalize(cls_token, p=2, dim=1)


class DinoV2Embedder:
    """Lazy wrapper around vanilla DINOv2 image embeddings."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        device: str = "auto",
        embedding_dim: int = 384,
    ) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.embedding_dim = embedding_dim

        self._initialized = False
        self._processor: Any = None
        self._model: Any = None

    @property
    def is_ready(self) -> bool:
        return self._initialized

    @property
    def backend_name(self) -> str:
        if not self._initialized:
            return "uninitialized"
        return "torch"

    def embed(self, image: str | Path | Image.Image | np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        prepared = self._prepare_image(image)
        return self._embed_torch(prepared)

    def warmup(self) -> None:
        self._ensure_loaded()

    def embed_batch(self, images: list[str | Path | Image.Image | np.ndarray], batch_size: int = 16) -> np.ndarray:
        self._ensure_loaded()
        batches: list[np.ndarray] = []

        for index in range(0, len(images), batch_size):
            chunk = [self._prepare_image(image) for image in images[index:index + batch_size]]
            tensors = [self._processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0) for image in chunk]
            pixel_values = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                outputs = self._model(pixel_values)
            batches.append(outputs.detach().cpu().numpy())

        if not batches:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.vstack(batches)

    def _ensure_loaded(self) -> None:
        if self._initialized:
            return

        if AutoImageProcessor is None:
            raise RuntimeError("Transformers dependencies are not installed.")

        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._load_torch_model()
        self._initialized = True

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_torch_model(self) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for model inference.")

        model = DinoV2BackboneEmbedder(model_name=self.model_name)
        model.to(self.device)
        model.eval()
        self._model = model

    def _prepare_image(self, image: str | Path | Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        return image

    def _embed_torch(self, image: Image.Image) -> np.ndarray:
        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            embedding = self._model(pixel_values)
        return embedding.detach().cpu().numpy().reshape(-1)

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return array / norms
