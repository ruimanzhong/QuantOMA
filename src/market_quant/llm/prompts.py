"""Prompt formatting for LLM gold event extraction."""

from __future__ import annotations

import json
from typing import Any


def format_event_prompt(template: str, title: str, summary: str) -> str:
    return template.format(title=title, summary=summary)


def build_event_messages(llm_config: dict[str, Any], title: str, summary: str) -> list[dict[str, str]]:
    user_prompt = format_event_prompt(llm_config["event_extraction_prompt"], title, summary)
    schema_instruction = llm_config.get("schema_instruction", "")
    if schema_instruction:
        user_prompt = f"{user_prompt}\n\n{schema_instruction}"

    messages = [{"role": "system", "content": llm_config["system_prompt"]}]
    for example in llm_config.get("few_shot_examples", []):
        messages.append(
            {
                "role": "user",
                "content": format_event_prompt(
                    llm_config["event_extraction_prompt"],
                    str(example["title"]),
                    str(example.get("summary", "")),
                ),
            }
        )
        messages.append({"role": "assistant", "content": json.dumps(example["output"], ensure_ascii=False)})
    messages.append({"role": "user", "content": user_prompt})
    return messages
