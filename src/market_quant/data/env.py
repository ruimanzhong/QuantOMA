"""Small environment loader for data clients."""

from __future__ import annotations

import os
from pathlib import Path


def load_env_files(project_root: str | Path, env_files: list[str] | None = None) -> None:
    root = Path(project_root)
    for env_file in env_files or [".env", "../gold_llm_quant/.env"]:
        path = Path(env_file).expanduser()
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
