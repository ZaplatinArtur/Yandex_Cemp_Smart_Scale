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

try:
    import timm
    from timm.data import create_transform, resolve_model_data_config
except ImportError:  # pragma: no cover - optional runtime deps
    timm = None
    create_transform = None
    resolve_model_data_config = None


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


class FineTunedTimmEmbedder(nn.Module if nn is not None else object):
    def __init__(self, checkpoint_path: str | Path) -> None:
        if torch is None or nn is None or F is None or timm is None:
            raise RuntimeError("PyTorch and timm dependencies are required for checkpoint inference.")

        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(self.checkpoint_path)

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise RuntimeError(f"Unsupported embedding checkpoint format: {self.checkpoint_path}")

        cfg = checkpoint.get("cfg") or {}
        state_dict = checkpoint["model"]
        self.model_name = str(cfg.get("timm_model", "vit_small_patch14_dinov2.lvd142m"))
        self.embedding_dim = int(cfg.get("embed_dim", 256))
        self.image_size = int(cfg.get("image_size", 224))

        self.backbone = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=0,
            img_size=self.image_size,
        )
        input_dim = int(getattr(self.backbone, "num_features", 0)) or int(state_dict["head.0.weight"].shape[1])
        hidden_dim = int(state_dict["head.0.weight"].shape[0])
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.embedding_dim),
        )

        backbone_state = {
            key.removeprefix("backbone."): value for key, value in state_dict.items() if key.startswith("backbone.")
        }
        head_state = {key.removeprefix("head."): value for key, value in state_dict.items() if key.startswith("head.")}
        self.backbone.load_state_dict(backbone_state, strict=True)
        self.head.load_state_dict(head_state, strict=True)

    def forward(self, pixel_values: Any) -> Any:
        features = self.backbone(pixel_values)
        embeddings = self.head(features)
        return F.normalize(embeddings, p=2, dim=1)


class DinoV2Embedder:
    """Lazy wrapper around vanilla DINOv2 image embeddings."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
        embedding_dim: int = 384,
    ) -> None:
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = self._resolve_device(device)
        self.embedding_dim = embedding_dim

        self._initialized = False
        self._processor: Any = None
        self._model: Any = None
        self._backend_name = "uninitialized"

    @property
    def is_ready(self) -> bool:
        return self._initialized

    @property
    def backend_name(self) -> str:
        if not self._initialized:
            return "uninitialized"
        return self._backend_name

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
            tensors = [self._image_to_tensor(image) for image in chunk]
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

        if self.checkpoint_path is not None:
            self._load_checkpoint_model()
        else:
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
        self._backend_name = "torch_transformers"

    def _load_checkpoint_model(self) -> None:
        if create_transform is None or resolve_model_data_config is None:
            raise RuntimeError("timm is required for embedding checkpoint inference.")

        model = FineTunedTimmEmbedder(self.checkpoint_path)
        if model.embedding_dim != self.embedding_dim:
            raise RuntimeError(
                f"Embedding checkpoint produces {model.embedding_dim} dimensions, "
                f"but SMART_SCALE_EMBEDDING_DIM is {self.embedding_dim}."
            )

        data_config = resolve_model_data_config(model.backbone)
        data_config["input_size"] = (3, model.image_size, model.image_size)
        self._processor = create_transform(**data_config, is_training=False)
        model.to(self.device)
        model.eval()
        self._model = model
        self.model_name = model.model_name
        self._backend_name = f"timm_checkpoint:{self.checkpoint_path.name}"

    def _prepare_image(self, image: str | Path | Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        return image

    def _embed_torch(self, image: Image.Image) -> np.ndarray:
        pixel_values = self._image_to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self._model(pixel_values)
        return embedding.detach().cpu().numpy().reshape(-1)

    def _image_to_tensor(self, image: Image.Image) -> Any:
        if self.checkpoint_path is not None:
            return self._processor(image)
        return self._processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return array / norms
