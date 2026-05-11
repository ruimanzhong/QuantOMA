"""OpenAI-compatible chat-completions client."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from urllib import request

from market_quant.llm.client.env import load_project_env


class LLMClient:
    """Minimal OpenAI-compatible client with sync and async JSON helpers."""

    def __init__(self, llm_config: dict[str, Any], project_root: str | Path | None = None):
        self.config = llm_config
        if project_root is not None:
            load_project_env(project_root, list(llm_config.get("env_files", [".env"])))

    @property
    def mock_mode(self) -> bool:
        return bool(self.config.get("mock_mode", True))

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Call a chat-completions endpoint and parse the assistant message as JSON."""
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
        timeout = float(self.config.get("request_timeout_seconds", 30))
        with request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)

    async def chat_json_async(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Async wrapper for concurrent batch extraction without extra HTTP dependencies."""
        return await asyncio.to_thread(self.chat_json, messages)
