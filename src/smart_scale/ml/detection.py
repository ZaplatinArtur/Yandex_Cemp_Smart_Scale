from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from smart_scale.domain import CropResult


class ProductLocalizer:
    """YOLO-based localization with a full-frame fallback."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.25,
        mask_threshold: float = 0.5,
        model: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self.mask_threshold = mask_threshold
        self._model: Any = model
        self._model_name = self.model_path.name if self.model_path else "full_frame_fallback"
        self._failure_reason: str | None = None

        if model is not None:
            self._model_name = getattr(model, "model_name", "injected_model")
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            self._failure_reason = "ultralytics_not_installed"
            return

        if self.model_path is None:
            self._failure_reason = "model_path_missing"
            return

        model_reference: str | None = None
        if self.model_path.exists():
            model_reference = str(self.model_path)
        elif self.model_path.name:
            model_reference = self.model_path.name

        if model_reference is None:
            self._failure_reason = "model_reference_missing"
            return

        try:
            self._model = YOLO(model_reference)
        except Exception as exc:
            self._model = None
            self._failure_reason = str(exc)

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def detector_name(self) -> str:
        return self._model_name

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    def localize(self, image: Image.Image) -> CropResult:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size

        if self._model is None:
            return self._full_frame(rgb_image, confidence=1.0)

        results = self._model.predict(np.asarray(rgb_image), conf=self.confidence_threshold, verbose=False)
        if not results:
            return self._full_frame(rgb_image, confidence=0.0)

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return self._full_frame(rgb_image, confidence=0.0)

        scores = result.boxes.conf.detach().cpu().numpy()
        best_idx = int(scores.argmax())
        confidence = float(scores[best_idx])
        bbox = self._best_bbox(result.boxes, best_idx, width, height)
        if result.masks is None or len(result.masks.data) <= best_idx:
            return self._crop_bbox(rgb_image, bbox=bbox, confidence=confidence)

        mask = result.masks.data[best_idx].detach().cpu().numpy()
        resized_mask = Image.fromarray(mask.astype(np.float32), mode="F").resize(rgb_image.size, Image.Resampling.NEAREST)
        binary_mask = np.asarray(resized_mask) > self.mask_threshold

        y_indices, x_indices = np.where(binary_mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return self._crop_bbox(rgb_image, bbox=bbox, confidence=confidence)

        rgb_array = np.asarray(rgb_image)
        isolated = np.zeros_like(rgb_array)
        isolated[binary_mask] = rgb_array[binary_mask]

        x1 = int(x_indices.min())
        x2 = int(x_indices.max()) + 1
        y1 = int(y_indices.min())
        y2 = int(y_indices.max()) + 1

        bbox = (x1, y1, x2, y2)
        cropped = Image.fromarray(isolated[y1:y2, x1:x2])
        return CropResult(
            image=cropped,
            bbox=bbox,
            confidence=confidence,
            detector_name=self._model_name,
            mask_applied=True,
        )

    def _best_bbox(self, boxes: Any, index: int, width: int, height: int) -> tuple[int, int, int, int] | None:
        xyxy = getattr(boxes, "xyxy", None)
        if xyxy is None:
            return None

        try:
            raw_bbox = xyxy.detach().cpu().numpy()[index]
        except Exception:
            return None

        x1, y1, x2, y2 = raw_bbox[:4]
        clamped = (
            max(0, min(width, int(round(float(x1))))),
            max(0, min(height, int(round(float(y1))))),
            max(0, min(width, int(round(float(x2))))),
            max(0, min(height, int(round(float(y2))))),
        )
        if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
            return None
        return clamped

    def _crop_bbox(
        self,
        image: Image.Image,
        *,
        bbox: tuple[int, int, int, int] | None,
        confidence: float,
    ) -> CropResult:
        if bbox is None:
            return self._full_frame(image, confidence=confidence)

        return CropResult(
            image=image.crop(bbox),
            bbox=bbox,
            confidence=confidence,
            detector_name=self._model_name,
            mask_applied=False,
        )

    def _full_frame(self, image: Image.Image, *, confidence: float) -> CropResult:
        width, height = image.size
        return CropResult(
            image=image,
            bbox=(0, 0, width, height),
            confidence=confidence,
            detector_name=self._model_name,
            mask_applied=False,
        )
