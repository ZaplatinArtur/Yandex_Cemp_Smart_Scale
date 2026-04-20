from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image


class CameraDevice(ABC):
    @abstractmethod
    def capture(self) -> Image.Image:
        raise NotImplementedError


class OpenCVCamera(CameraDevice):
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index

    def capture(self) -> Image.Image:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("opencv-python не установлен.") from exc

        stream = cv2.VideoCapture(self.camera_index)
        ok, frame = stream.read()
        stream.release()
        if not ok:
            raise RuntimeError("Не удалось получить кадр с камеры.")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


class MockCamera(CameraDevice):
    def __init__(self, image_path: str | Path) -> None:
        self.image_path = Path(image_path)

    def capture(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")
