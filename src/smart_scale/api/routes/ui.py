from __future__ import annotations

import io
import csv
from datetime import datetime, timezone
from pathlib import Path
import hmac
import json
import mimetypes
import os
import logging
import re
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request, Depends, HTTPException, Query, Body, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

from smart_scale.api.dependencies import get_pipeline
from smart_scale.ml.catalog_seed import SUPPORTED_IMAGE_EXTENSIONS, load_price_catalog, normalize_catalog_label, split_sort_label
from smart_scale.ml.pipeline import RecognitionPipeline


ui_router = APIRouter()
LOGGER = logging.getLogger("smart_scale.api.ui")

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


@ui_router.get("/ui", response_class=HTMLResponse)
async def ui_index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "ui.html", {"api_prefix": "/api"})


@ui_router.get("/api/catalog/varieties")
async def catalog_varieties(pipeline: RecognitionPipeline = Depends(get_pipeline)) -> JSONResponse:
    items = _build_catalog_items(pipeline.settings)
    return JSONResponse({"items": items, "count": len(items)})


@ui_router.get("/api/predictions/latest")
async def latest_prediction(request: Request) -> JSONResponse:
    history = getattr(request.app.state, "prediction_history", None)
    if history is None:
        return JSONResponse({"status": "empty", "prediction": None})
    record = history.latest()
    if record is None:
        return JSONResponse({"status": "empty", "prediction": None})
    return JSONResponse({"status": "ok", **record})


@ui_router.get("/api/predictions/{prediction_id}/image")
async def serve_prediction_image(request: Request, prediction_id: str) -> Response:
    history = getattr(request.app.state, "prediction_history", None)
    if history is None:
        raise HTTPException(status_code=404, detail="Prediction history is empty")

    record = history.get(prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    image_path = _resolve_history_file(record.get("image_path"), history.storage_dir.resolve())
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Prediction image not found")

    content_type = str(record.get("content_type") or "")
    mime, _ = mimetypes.guess_type(str(image_path))
    media_type = content_type if content_type.startswith("image/") else mime or "image/jpeg"
    return FileResponse(str(image_path), media_type=media_type, headers={"Cache-Control": "no-store"})


@ui_router.get("/api/predictions/{prediction_id}/crop")
async def serve_prediction_crop(request: Request, prediction_id: str) -> Response:
    history = getattr(request.app.state, "prediction_history", None)
    if history is None:
        raise HTTPException(status_code=404, detail="Prediction history is empty")

    record = history.get(prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    storage_root = history.storage_dir.resolve()
    crop_image_path = record.get("crop_image_path")
    if crop_image_path:
        resolved_crop_path = _resolve_history_file(crop_image_path, storage_root)
        if resolved_crop_path.exists() and resolved_crop_path.is_file():
            return FileResponse(str(resolved_crop_path), media_type="image/jpeg")

    bbox = _prediction_bbox(record)
    if bbox is None:
        raise HTTPException(status_code=404, detail="Prediction crop not found")

    image_path = _resolve_history_file(record.get("image_path"), storage_root)
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Prediction image not found")

    try:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            clamped_bbox = _clamp_bbox(bbox, rgb_image.size)
            if clamped_bbox is None:
                raise HTTPException(status_code=400, detail="Invalid prediction crop bbox")
            cropped = rgb_image.crop(clamped_bbox)
            buffer = io.BytesIO()
            cropped.save(buffer, format="JPEG", quality=90)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Prediction image is not readable") from exc
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Prediction image not found") from exc

    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@ui_router.post("/api/feedback/incorrect")
async def record_incorrect_feedback(
    request: Request,
    image: UploadFile | None = File(None),
    correct_label: str = Form(...),
    prediction_id: str | None = Form(None),
    prediction_json: str = Form("{}"),
    selected_from: str = Form("catalog"),
    pipeline: RecognitionPipeline = Depends(get_pipeline),
) -> JSONResponse:
    settings = pipeline.settings
    catalog_items = _build_catalog_items(settings)
    catalog_by_label = {item["label"]: item for item in catalog_items}
    normalized_label = normalize_catalog_label(correct_label)
    if normalized_label not in catalog_by_label:
        raise HTTPException(status_code=400, detail="Unknown product variety")

    payload: bytes
    original_filename: str | None
    content_type: str | None
    stored_record = None
    if image is not None:
        payload = await image.read()
        original_filename = image.filename
        content_type = image.content_type
    elif prediction_id:
        history = getattr(request.app.state, "prediction_history", None)
        stored_record = history.get(prediction_id) if history is not None else None
        if stored_record is None:
            raise HTTPException(status_code=404, detail="Prediction image not found")
        stored_image_path = Path(str(stored_record["image_path"]))
        try:
            payload = stored_image_path.read_bytes()
        except OSError as exc:
            raise HTTPException(status_code=404, detail="Prediction image not found") from exc
        original_filename = stored_record.get("original_filename")
        content_type = stored_record.get("content_type")
    else:
        raise HTTPException(status_code=400, detail="Image or prediction_id is required")

    if not payload:
        raise HTTPException(status_code=400, detail="Image file is empty")
    if content_type and not str(content_type).startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected image file")

    try:
        prediction_payload = json.loads(prediction_json) if prediction_json else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid prediction_json") from exc
    if not isinstance(prediction_payload, dict):
        raise HTTPException(status_code=400, detail="prediction_json must be an object")
    if not prediction_payload and stored_record is not None:
        prediction_payload = dict(stored_record.get("prediction") or {})

    target_dir = settings.feedback_dir / normalized_label
    target_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(original_filename or "").suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        suffix = ".jpg"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"{timestamp}_{uuid4().hex[:12]}"
    image_path = target_dir / f"{stem}{suffix}"
    metadata_path = target_dir / f"{stem}.json"

    image_path.write_bytes(payload)
    catalog_item = catalog_by_label[normalized_label]
    metadata = {
        "correct_label": normalized_label,
        "product_type": catalog_item["product_type"],
        "product_sort": catalog_item["product_sort"],
        "selected_from": selected_from,
        "prediction_id": prediction_id,
        "original_filename": original_filename,
        "content_type": content_type,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "prediction": prediction_payload,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info(
        "event=incorrect_feedback_saved label=%s selected_from=%s image_path=%s",
        normalized_label,
        selected_from,
        image_path,
    )
    return JSONResponse(
        {
            "status": "ok",
            "label": normalized_label,
            "image_path": str(image_path),
            "metadata_path": str(metadata_path),
        }
    )


@ui_router.post("/api/admin/catalog/examples")
async def add_admin_catalog_example(
    request: Request,
    image: UploadFile | None = File(None),
    admin_token: str = Form(...),
    product_type: str = Form(...),
    product_sort: str = Form(...),
    price_rub_per_kg: float = Form(...),
    prediction_id: str | None = Form(None),
    pipeline: RecognitionPipeline = Depends(get_pipeline),
) -> JSONResponse:
    settings = pipeline.settings
    _verify_admin_token(settings.admin_token, admin_token)

    normalized_type = _normalize_slug(product_type, "product_type", r"[a-z0-9]+")
    normalized_sort = _normalize_slug(product_sort, "product_sort", r"[a-z0-9_]+")
    if price_rub_per_kg <= 0:
        raise HTTPException(status_code=400, detail="price_rub_per_kg must be greater than zero")

    label = normalize_catalog_label(f"{normalized_type}_{normalized_sort}")
    catalog_prices = _load_price_items(settings.price_catalog_path)
    existing_price = catalog_prices.get(label)
    if existing_price is not None and abs(float(existing_price) - float(price_rub_per_kg)) > 0.005:
        raise HTTPException(status_code=409, detail="Price for this product variety already exists and differs")

    payload, original_filename, content_type = await _read_catalog_example_payload(
        request,
        image=image,
        prediction_id=prediction_id,
    )
    pil_image = _load_rgb_image(payload, content_type=content_type)

    train_dir = _resolve_train_dataset_dir(settings)
    target_dir = train_dir / label
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"{timestamp}_{uuid4().hex[:12]}"
    image_path = target_dir / f"{stem}.jpg"
    pil_image.save(image_path, format="JPEG", quality=95)

    product_id = f"{label}:{stem}"
    try:
        match = await run_in_threadpool(
            pipeline.add_catalog_example,
            pil_image,
            product_type=normalized_type,
            product_sort=normalized_sort,
            price_rub_per_kg=float(price_rub_per_kg),
            product_id=product_id,
            image_path=image_path,
        )
    except Exception as exc:
        try:
            image_path.unlink()
        except OSError:
            pass
        raise HTTPException(status_code=503, detail="Failed to add catalog embedding") from exc

    catalog_updated = existing_price is None
    if catalog_updated:
        _upsert_price_catalog(settings.price_catalog_path, label, float(price_rub_per_kg))

    LOGGER.info(
        "event=admin_catalog_example_added label=%s product_id=%s image_path=%s catalog_updated=%s original_filename=%s",
        label,
        match.product_id,
        image_path,
        catalog_updated,
        original_filename,
    )
    return JSONResponse(
        {
            "status": "ok",
            "label": label,
            "product_id": match.product_id,
            "product_type": normalized_type,
            "product_sort": normalized_sort,
            "price_rub_per_kg": float(price_rub_per_kg),
            "image_path": str(image_path),
            "catalog_updated": catalog_updated,
        }
    )


@ui_router.get("/api/serve_image")
async def serve_image(p: str = Query(...), pipeline: RecognitionPipeline = Depends(get_pipeline)) -> FileResponse:
    """Serve an image file from the catalog by absolute or relative path.

    The UI will call this endpoint with the `metadata.path` returned by `/api/predict`.
    For safety, the resolved path must be located inside the settings.image_catalog_dir.
    """
    settings = pipeline.settings
    path = Path(p)
    if not path.is_absolute():
        path = settings.image_catalog_dir / path

    try:
        resolved = path.resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image path")

    allowed_roots = [settings.image_catalog_dir.resolve(), settings.dataset_dir.resolve()]
    if _path_is_inside_any(resolved, allowed_roots):
        if not resolved.exists() or not resolved.is_file():
            LOGGER.warning("serve_image: file not found %s", resolved)
            raise HTTPException(status_code=404, detail="Image not found")
        mime, _ = mimetypes.guess_type(str(resolved))
        return FileResponse(str(resolved), media_type=mime or "application/octet-stream")

    images_root = settings.image_catalog_dir.resolve()
    try:
        # Use os.path.commonpath for a robust containment check across platforms
        common = os.path.commonpath([str(images_root), str(resolved)])
        if common != str(images_root):
            # Requested path sits outside the image root. Try to fallback by filename
            logging.getLogger("smart_scale.api.ui").info(
                "serve_image: requested %s not inside %s — attempting filename fallback",
                resolved,
                images_root,
            )
            filename = resolved.name
            matches = list(images_root.rglob(filename))
            if matches:
                resolved = matches[0].resolve()
                logging.getLogger("smart_scale.api.ui").info("serve_image: fallback to %s", resolved)
            else:
                logging.getLogger("smart_scale.api.ui").warning(
                    "serve_image: access denied for %s (not inside %s) and no fallback found",
                    resolved,
                    images_root,
                )
                raise HTTPException(status_code=403, detail="Access denied")
    except ValueError:
        # Occurs if paths are on different drives on Windows — attempt filename fallback
        logging.getLogger("smart_scale.api.ui").info(
            "serve_image: path on different drive %s vs %s — attempting filename fallback",
            resolved,
            images_root,
        )
        filename = resolved.name
        matches = list(images_root.rglob(filename))
        if matches:
            resolved = matches[0].resolve()
            logging.getLogger("smart_scale.api.ui").info("serve_image: fallback to %s", resolved)
        else:
            logging.getLogger("smart_scale.api.ui").warning(
                "serve_image: path on different drive %s vs %s and no fallback found",
                resolved,
                images_root,
            )
            raise HTTPException(status_code=403, detail="Access denied")

    if not resolved.exists() or not resolved.is_file():
        logging.getLogger("smart_scale.api.ui").warning("serve_image: file not found %s", resolved)
        raise HTTPException(status_code=404, detail="Image not found")

    mime, _ = mimetypes.guess_type(str(resolved))
    return FileResponse(str(resolved), media_type=mime or "application/octet-stream")


@ui_router.post("/api/selection")
async def record_selection(selection: dict = Body(...), pipeline: RecognitionPipeline = Depends(get_pipeline)) -> JSONResponse:
    """Receive user selection from the UI. Body example: {"selected": "product_id"} or {"selected": "none"}

    Currently this endpoint only acknowledges the choice; it can be extended to log or persist feedback.
    """
    selected = selection.get("selected") if isinstance(selection, dict) else None
    # Placeholder: in future persist user feedback via pipeline or storage
    return JSONResponse({"status": "ok", "selected": selected})


def _verify_admin_token(configured_token: str | None, provided_token: str) -> None:
    if not configured_token:
        raise HTTPException(status_code=403, detail="Пароль администратора не настроен")
    if not hmac.compare_digest(str(configured_token), str(provided_token)):
        raise HTTPException(status_code=403, detail="Неверный пароль администратора")


def _normalize_slug(value: str, field_name: str, pattern: str) -> str:
    normalized = str(value or "").strip().lower()
    if not re.fullmatch(pattern, normalized):
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}")
    return normalized


async def _read_catalog_example_payload(
    request: Request,
    *,
    image: UploadFile | None,
    prediction_id: str | None,
) -> tuple[bytes, str | None, str | None]:
    if image is not None:
        payload = await image.read()
        original_filename = image.filename
        content_type = image.content_type
    elif prediction_id:
        history = getattr(request.app.state, "prediction_history", None)
        stored_record = history.get(prediction_id) if history is not None else None
        if stored_record is None:
            raise HTTPException(status_code=404, detail="Prediction image not found")

        image_path = _resolve_history_file(stored_record.get("image_path"), history.storage_dir.resolve())
        try:
            payload = image_path.read_bytes()
        except OSError as exc:
            raise HTTPException(status_code=404, detail="Prediction image not found") from exc
        original_filename = stored_record.get("original_filename")
        content_type = stored_record.get("content_type")
    else:
        raise HTTPException(status_code=400, detail="Image or prediction_id is required")

    if not payload:
        raise HTTPException(status_code=400, detail="Image file is empty")
    if content_type and not str(content_type).startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected image file")
    return payload, original_filename, content_type


def _load_rgb_image(payload: bytes, *, content_type: str | None) -> Image.Image:
    if content_type and not str(content_type).startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected image file")
    try:
        return Image.open(io.BytesIO(payload)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Could not recognize image") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Could not process image") from exc


def _resolve_train_dataset_dir(settings) -> Path:
    dataset_dir = settings.dataset_dir
    if dataset_dir.name.lower() == "train":
        return dataset_dir
    if (dataset_dir / "train").exists() or (dataset_dir / "test").exists():
        return dataset_dir / "train"
    return dataset_dir


def _upsert_price_catalog(path: Path, label: str, price_rub_per_kg: float) -> None:
    prices = _load_price_items(path)
    prices[normalize_catalog_label(label)] = float(price_rub_per_kg)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = {key: prices[key] for key in sorted(prices)}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    if suffix == ".csv":
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["label", "price_rub_per_kg"])
            writer.writeheader()
            for key in sorted(prices):
                writer.writerow({"label": key, "price_rub_per_kg": _format_price_literal(prices[key])})
        return

    if suffix in {"", ".py"}:
        path.write_text(_format_python_price_catalog(prices), encoding="utf-8")
        return

    raise HTTPException(status_code=400, detail=f"Unsupported price catalog format: {path.suffix}")


def _format_python_price_catalog(prices: dict[str, float]) -> str:
    lines = ["prices = {"]
    for label in sorted(prices):
        lines.append(f'    "{label}": {_format_price_literal(prices[label])},')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _format_price_literal(value: float) -> str:
    return f"{float(value):.2f}"


def _build_catalog_items(settings) -> list[dict[str, object]]:
    records: dict[str, dict[str, object]] = {}

    for label, price in _load_price_items(settings.price_catalog_path).items():
        product_type, product_sort = split_sort_label(label)
        records[label] = {
            "label": label,
            "product_type": product_type,
            "product_sort": product_sort,
            "price_rub_per_kg": price,
            "sources": ["price"],
        }

    for label in _discover_dataset_labels(settings.dataset_dir):
        product_type, product_sort = split_sort_label(label)
        record = records.setdefault(
            label,
            {
                "label": label,
                "product_type": product_type,
                "product_sort": product_sort,
                "price_rub_per_kg": None,
                "sources": [],
            },
        )
        sources = record.setdefault("sources", [])
        if isinstance(sources, list) and "dataset" not in sources:
            sources.append("dataset")

    return sorted(records.values(), key=lambda item: str(item["label"]))


def _load_price_items(path: Path) -> dict[str, float]:
    try:
        return load_price_catalog(path)
    except FileNotFoundError:
        logging.getLogger("smart_scale.api.ui").warning("event=price_catalog_missing path=%s", path)
        return {}


def _discover_dataset_labels(dataset_dir: Path) -> set[str]:
    if not dataset_dir.exists():
        return set()

    labels: set[str] = set()
    for path in dataset_dir.rglob("*"):
        if not path.is_dir():
            continue
        label = normalize_catalog_label(path.name)
        if _looks_like_product_label(label):
            labels.add(label)
    return labels


def _looks_like_product_label(label: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)+", label))


def _path_is_inside_any(path: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            if os.path.commonpath([str(root), str(path)]) == str(root):
                return True
        except ValueError:
            continue
    return False


def _resolve_history_file(path_value: Any, storage_root: Path) -> Path:
    if not path_value:
        raise HTTPException(status_code=404, detail="Prediction image not found")

    path = Path(str(path_value))
    if not path.is_absolute():
        path = storage_root / path

    try:
        resolved = path.resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid prediction image path") from exc

    if not _path_is_inside_any(resolved, [storage_root]):
        raise HTTPException(status_code=403, detail="Access denied")
    return resolved


def _prediction_bbox(record: dict[str, Any]) -> tuple[int, int, int, int] | None:
    prediction = record.get("prediction")
    if not isinstance(prediction, dict):
        return None

    crop = prediction.get("crop")
    if not isinstance(crop, dict):
        return None

    bbox = crop.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        return tuple(int(round(float(value))) for value in bbox)
    except (TypeError, ValueError):
        return None


def _clamp_bbox(bbox: tuple[int, int, int, int], image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    clamped = (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        return None
    return clamped
