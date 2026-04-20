from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from smart_scale.domain import AnomalyCheckResult


class HandAnomalyDetector:
    """MediaPipe-based check that blocks inference when hands are present."""

    def __init__(
        self,
        enabled: bool = True,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        model_asset_path: str | Path | None = None,
        detector: Any | None = None,
    ) -> None:
        self.enabled = enabled
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.model_asset_path = Path(model_asset_path) if model_asset_path else None
        self._detector: Any = detector
        self._mp: Any = None
        self._is_available = False
        self._skip_reason: str | None = None

        if not enabled:
            self._skip_reason = "disabled"
            return

        if detector is not None:
            self._is_available = True
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError:
            self._skip_reason = "dependency_missing"
            return

        if self.model_asset_path is None or not self.model_asset_path.exists():
            self._skip_reason = "model_asset_missing"
            return

        base_options = python.BaseOptions(model_asset_path=str(self.model_asset_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._mp = mp
        self._is_available = True

    @property
    def is_ready(self) -> bool:
        return self._is_available and self._detector is not None

    @property
    def skip_reason(self) -> str | None:
        return self._skip_reason

    def detect(self, image: Any) -> AnomalyCheckResult:
        if not self.enabled:
            return AnomalyCheckResult(blocked=False, message="Проверка рук отключена.")

        if not self._is_available or self._detector is None:
            return AnomalyCheckResult(
                blocked=False,
                message="MediaPipe недоступен, этап детекции рук пропущен.",
                warning_code="hand_check_skipped",
            )

        rgb_frame = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._detector.detect(mp_image)
        hands_count = len(result.hand_landmarks or [])

        if hands_count > 0:
            return AnomalyCheckResult(
                blocked=True,
                message="Обнаружены руки на платформе весов. Инференс остановлен.",
                hands_count=hands_count,
                warning_code="hands_detected",
            )

        return AnomalyCheckResult(
            blocked=False,
            message="Аномалий не обнаружено.",
            hands_count=0,
        )

    def close(self) -> None:
        if self._detector is not None and hasattr(self._detector, "close"):
            self._detector.close()
