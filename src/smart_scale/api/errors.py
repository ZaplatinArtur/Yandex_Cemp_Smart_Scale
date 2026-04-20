from __future__ import annotations


class ServiceUnavailableError(RuntimeError):
    """Raised when the inference service cannot serve a valid request."""

    def __init__(self, detail: str = "Сервис инференса временно недоступен.") -> None:
        super().__init__(detail)
        self.detail = detail
