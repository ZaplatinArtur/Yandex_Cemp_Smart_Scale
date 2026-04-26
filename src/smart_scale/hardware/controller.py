from __future__ import annotations

import time
from collections import deque

from smart_scale.domain import CaptureBundle, WeightSnapshot
from smart_scale.hardware.camera import CameraDevice
from smart_scale.hardware.scale import ScaleReader


class SmartScaleController:
    """Client-side controller for weight stabilization and photo capture."""

    def __init__(
        self,
        camera: CameraDevice,
        scale_reader: ScaleReader,
        tolerance_grams: float = 2.0,
        stable_window: int = 5,
        polling_interval_sec: float = 0.2,
    ) -> None:
        self.camera = camera
        self.scale_reader = scale_reader
        self.tolerance_grams = tolerance_grams
        self.stable_window = stable_window
        self.polling_interval_sec = polling_interval_sec

    def wait_for_stable_weight(self) -> WeightSnapshot:
        samples: deque[float] = deque(maxlen=self.stable_window)
        while True:
            current_weight = float(self.scale_reader.read_weight_grams())
            samples.append(current_weight)
            if len(samples) < self.stable_window:
                time.sleep(self.polling_interval_sec)
                continue

            spread = max(samples) - min(samples)
            average_weight = sum(samples) / len(samples)
            if average_weight > 0 and spread <= self.tolerance_grams:
                return WeightSnapshot(
                    weight_grams=round(average_weight, 3),
                    stable=True,
                    samples=list(samples),
                    spread=round(spread, 3),
                )

            time.sleep(self.polling_interval_sec)

    def capture_bundle(self) -> CaptureBundle:
        stable_weight = self.wait_for_stable_weight()
        frame = self.camera.capture()
        return CaptureBundle(image=frame, weight=stable_weight)
