from __future__ import annotations

from pathlib import Path
import mimetypes
import os
import logging

from fastapi import APIRouter, Request, Depends, HTTPException, Query, Body
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates

from smart_scale.api.dependencies import get_pipeline
from smart_scale.ml.pipeline import RecognitionPipeline


ui_router = APIRouter()

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


@ui_router.get("/ui", response_class=HTMLResponse)
async def ui_index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("ui.html", {"request": request, "api_prefix": "/api"})


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
