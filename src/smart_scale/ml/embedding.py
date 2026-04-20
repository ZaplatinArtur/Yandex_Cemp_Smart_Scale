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
    from torchvision import transforms
    from transformers import AutoImageProcessor, AutoModel
except ImportError:  # pragma: no cover - optional runtime deps
    torch = None
    nn = None
    F = None
    transforms = None
    AutoImageProcessor = None
    AutoModel = None


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class ProjectionHeadEmbedder(nn.Module if nn is not None else object):
    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-small",
        backbone_dim: int = 384,
        embedding_dim: int = 256,
    ) -> None:
        if nn is None or AutoModel is None:
            raise RuntimeError("PyTorch/Transformers dependencies are not installed.")

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

    def forward(self, pixel_values: Any) -> Any:
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        embedding = self.head(cls_token)
        return F.normalize(embedding, p=2, dim=1)


class DinoV2Embedder:
    """Lazy wrapper around the fine-tuned DINOv2 checkpoint used in the project."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        onnx_model_path: str | Path | None = None,
        device: str = "auto",
        embedding_dim: int = 256,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.onnx_model_path = Path(onnx_model_path) if onnx_model_path else None
        self.device = self._resolve_device(device)
        self.embedding_dim = embedding_dim

        self._initialized = False
        self._use_onnx = False
        self._checkpoint: dict[str, Any] = {}
        self._processor: Any = None
        self._preprocess: Any = None
        self._model: Any = None
        self._onnx_session: Any = None
        self._onnx_input_name: str | None = None

    @property
    def is_ready(self) -> bool:
        return self._initialized

    @property
    def backend_name(self) -> str:
        if not self._initialized:
            return "uninitialized"
        return "onnx" if self._use_onnx else "torch"

    def embed(self, image: str | Path | Image.Image | np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        prepared = self._prepare_image(image)
        if self._use_onnx:
            return self._embed_onnx(prepared)
        return self._embed_torch(prepared)

    def warmup(self) -> None:
        self._ensure_loaded()

    def embed_batch(self, images: list[str | Path | Image.Image | np.ndarray], batch_size: int = 16) -> np.ndarray:
        self._ensure_loaded()
        batches: list[np.ndarray] = []

        for index in range(0, len(images), batch_size):
            chunk = [self._prepare_image(image) for image in images[index:index + batch_size]]
            if self._use_onnx:
                arrays = [self._processor(images=image, return_tensors="np")["pixel_values"].squeeze(0) for image in chunk]
                batch = np.ascontiguousarray(np.stack(arrays).astype(np.float32))
                outputs = self._onnx_session.run(None, {self._onnx_input_name: batch})[0]
                batches.append(self._normalize(outputs))
                continue

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

        if AutoImageProcessor is None or transforms is None:
            raise RuntimeError("Transformers/Torchvision dependencies are not installed.")

        checkpoint = self._load_checkpoint()
        backbone_name = checkpoint.get("backbone_name", "facebook/dinov2-small")
        self.embedding_dim = int(checkpoint.get("embedding_dim", self.embedding_dim))
        self._processor = AutoImageProcessor.from_pretrained(backbone_name)
        self._preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
        )

        if self._try_load_onnx():
            self._initialized = True
            self._use_onnx = True
            return

        self._load_torch_model(backbone_name)
        self._initialized = True
        self._use_onnx = False

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_checkpoint(self) -> dict[str, Any]:
        if not self.checkpoint_path.exists() or self.checkpoint_path.suffix.lower() == ".onnx":
            self._checkpoint = {}
            if self.checkpoint_path.suffix.lower() == ".onnx" and self.onnx_model_path is None:
                self.onnx_model_path = self.checkpoint_path
            return self._checkpoint

        if torch is None:
            raise RuntimeError("PyTorch is required to load the checkpoint.")

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        self._checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        return self._checkpoint

    def _try_load_onnx(self) -> bool:
        if self.onnx_model_path is None or not self.onnx_model_path.exists():
            return False

        try:
            import onnxruntime as ort
        except ImportError:
            return False

        providers = ["CPUExecutionProvider"]
        if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._onnx_session = ort.InferenceSession(str(self.onnx_model_path), providers=providers)
        self._onnx_input_name = self._onnx_session.get_inputs()[0].name
        return True

    def _load_torch_model(self, backbone_name: str) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for model inference.")

        model = ProjectionHeadEmbedder(backbone_name=backbone_name, embedding_dim=self.embedding_dim)
        checkpoint = self._checkpoint

        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif any(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            model.load_state_dict(checkpoint, strict=False)

        if "head_state_dict" in checkpoint:
            model.head.load_state_dict(checkpoint["head_state_dict"], strict=False)
        if "backbone_state_dict" in checkpoint:
            model.backbone.load_state_dict(checkpoint["backbone_state_dict"], strict=False)

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

        return self._preprocess(image)

    def _embed_onnx(self, image: Image.Image) -> np.ndarray:
        inputs = self._processor(images=image, return_tensors="np")
        pixel_values = np.ascontiguousarray(inputs["pixel_values"].astype(np.float32))
        outputs = self._onnx_session.run(None, {self._onnx_input_name: pixel_values})[0]
        return self._normalize(outputs).reshape(-1)

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
