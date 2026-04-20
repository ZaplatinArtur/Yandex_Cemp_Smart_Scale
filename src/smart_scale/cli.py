from __future__ import annotations

import uvicorn

from smart_scale.api import app
from smart_scale.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
