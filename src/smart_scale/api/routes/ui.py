from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import mimetypes
import os
import logging
import re
from uuid import uuid4

from fastapi import APIRouter, Request, Depends, HTTPException, Query, Body, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates

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
