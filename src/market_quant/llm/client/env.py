"""Environment helpers for LLM clients."""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(env_path: str | Path) -> None:
    """Load simple KEY=VALUE pairs from .env without adding a runtime dependency."""
    env_path = Path(env_path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_project_env(project_root: str | Path, env_files: list[str] | None = None) -> None:
    root = Path(project_root)
    for env_file in env_files or [".env"]:
        path = Path(env_file).expanduser()
        if not path.is_absolute():
            path = root / path
        load_env_file(path)
