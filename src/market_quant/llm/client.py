"""Small OpenAI-compatible LLM client for structured event extraction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib import request


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


class LLMClient:
    """Minimal chat-completions client using the standard library."""

    def __init__(self, llm_config: dict[str, Any], project_root: str | Path | None = None):
        self.config = llm_config
        if project_root is not None:
            load_project_env(project_root, list(llm_config.get("env_files", [".env"])))

    @property
    def mock_mode(self) -> bool:
        return bool(self.config.get("mock_mode", True))

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        if self.mock_mode:
            raise RuntimeError("LLMClient.chat_json should not be called in mock_mode")
        api_key_env = self.config["api_key_env"]
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key env var: {api_key_env}")
        payload = {
            "model": self.config["model"],
            "temperature": float(self.config.get("temperature", 0.0)),
            "messages": messages,
        }
        body = json.dumps(payload).encode("utf-8")
        url = f"{self.config['base_url'].rstrip('/')}/v1/chat/completions"
        req = request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=float(self.config.get("request_timeout_seconds", 30))) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
