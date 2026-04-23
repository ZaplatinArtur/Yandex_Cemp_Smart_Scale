from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from smart_scale.ml.catalog_seed import SUPPORTED_IMAGE_EXTENSIONS


class PredictionHistory:
    def __init__(self, storage_dir: Path, max_items: int = 50) -> None:
        self.storage_dir = storage_dir
        self.max_items = max_items
        self._records: deque[dict[str, Any]] = deque(maxlen=max_items)
        self._lock = Lock()

    def record(
        self,
        prediction: Any,
        image_payload: bytes,
        *,
        filename: str | None,
        content_type: str | None,
        top_k: int,
    ) -> dict[str, Any]:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        prediction_id = uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        suffix = Path(filename or "").suffix.lower()
        if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            suffix = ".jpg"
        image_path = self.storage_dir / f"{prediction_id}{suffix}"
        image_path.write_bytes(image_payload)

        prediction_payload = _dump_model(prediction)
        prediction_payload["prediction_id"] = prediction_id
        prediction_payload["created_at"] = created_at

        record = {
            "prediction_id": prediction_id,
            "created_at": created_at,
            "original_filename": filename,
            "content_type": content_type,
            "image_path": str(image_path),
            "top_k": top_k,
            "prediction": prediction_payload,
        }
        with self._lock:
            self._records.appendleft(record)
        return record

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._records[0]) if self._records else None

    def get(self, prediction_id: str) -> dict[str, Any] | None:
        with self._lock:
            for record in self._records:
                if record.get("prediction_id") == prediction_id:
                    return dict(record)
        return None


def _dump_model(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)
