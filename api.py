from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_dir_on_path() -> None:
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


try:
    from smart_scale.api import app
    from smart_scale.cli import main
except ModuleNotFoundError:
    _ensure_src_dir_on_path()
    from smart_scale.api import app
    from smart_scale.cli import main


if __name__ == "__main__":
    main()
