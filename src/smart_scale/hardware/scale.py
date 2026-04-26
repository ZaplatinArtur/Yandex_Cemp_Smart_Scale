from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class ScaleReader(ABC):
    @abstractmethod
    def read_weight_grams(self) -> float:
        raise NotImplementedError


class SerialScaleReader(ScaleReader):
    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

    def read_weight_grams(self) -> float:
        try:
            import serial
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("pyserial не установлен.") from exc

        with serial.Serial(self.port, self.baudrate, timeout=self.timeout) as connection:
            raw = connection.readline().decode("utf-8", errors="ignore").strip()
        return float(raw.replace(",", "."))


class MockScaleReader(ScaleReader):
    def __init__(self, values: Iterable[float] | None = None, fallback_value: float = 0.0) -> None:
        self.values = iter(values or [])
        self.fallback_value = fallback_value
        self._last_value = fallback_value

    def read_weight_grams(self) -> float:
        try:
            self._last_value = float(next(self.values))
        except StopIteration:
            pass
        return self._last_value
