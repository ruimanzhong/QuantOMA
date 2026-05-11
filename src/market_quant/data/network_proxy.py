"""Scoped proxy helpers for data-source-specific network access."""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Any, Iterator


PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


def _env_value(config: dict[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value:
        return str(value)
    env_name = config.get(f"{key}_env")
    if env_name:
        return os.getenv(str(env_name))
    return None


def build_proxy_environment(config: dict[str, Any] | None) -> dict[str, str]:
    """Build proxy environment variables from config and env var references."""
    proxy_cfg = (config or {}).get("proxy", config or {})
    if not proxy_cfg.get("enabled", False):
        return {}

    default_proxy = _env_value(proxy_cfg, "proxy_url") or os.getenv(str(proxy_cfg.get("proxy_env", "")))
    http_proxy = _env_value(proxy_cfg, "http_proxy_url") or os.getenv(str(proxy_cfg.get("http_proxy_env", ""))) or default_proxy
    https_proxy = _env_value(proxy_cfg, "https_proxy_url") or os.getenv(str(proxy_cfg.get("https_proxy_env", ""))) or default_proxy
    all_proxy = _env_value(proxy_cfg, "all_proxy_url") or os.getenv(str(proxy_cfg.get("all_proxy_env", "")))
    no_proxy = _env_value(proxy_cfg, "no_proxy") or os.getenv(str(proxy_cfg.get("no_proxy_env", "")))

    env: dict[str, str] = {}
    if http_proxy:
        env["HTTP_PROXY"] = http_proxy
        env["http_proxy"] = http_proxy
    if https_proxy:
        env["HTTPS_PROXY"] = https_proxy
        env["https_proxy"] = https_proxy
    if all_proxy:
        env["ALL_PROXY"] = all_proxy
        env["all_proxy"] = all_proxy
    if no_proxy:
        env["NO_PROXY"] = no_proxy
        env["no_proxy"] = no_proxy
    return env


@contextmanager
def scoped_proxy_environment(config: dict[str, Any] | None, label: str = "network") -> Iterator[dict[str, str]]:
    """Temporarily apply proxy env vars, then restore the previous process env."""
    proxy_env = build_proxy_environment(config)
    previous = {key: os.environ.get(key) for key in PROXY_ENV_KEYS}
    try:
        if proxy_env:
            os.environ.update(proxy_env)
            print(f"using scoped proxy for {label}: {', '.join(sorted(proxy_env))}")
        elif (config or {}).get("proxy", config or {}).get("enabled", False):
            print(f"warning: scoped proxy enabled for {label}, but no proxy URL env/value is configured")
        yield proxy_env
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
