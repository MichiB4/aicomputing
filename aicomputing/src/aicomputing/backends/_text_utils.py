"""
Shared text helpers for backends: thinking-trace stripping and the
/no_think toggle. Used by both the Ollama and OpenAI-compatible backends
so the behaviour stays identical across providers.
"""

from __future__ import annotations

import re
from typing import Dict, List


def append_no_think(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Append /no_think to the last user message to disable Qwen3 thinking."""
    messages = [dict(m) for m in messages]
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if "/no_think" not in content:
                msg["content"] = content.rstrip() + "\n/no_think"
            break
    return messages


def strip_thinking(text: str) -> str:
    """
    Strip visible <think>...</think> blocks from model output.

    Some providers (OpenRouter, SambaNova) and local Ollama models return
    the reasoning trace in the message content rather than a separate
    field. We strip it to keep raw_text clean for downstream parsing.

    If the answer is truncated mid-thought (nothing after </think>, or no
    closing tag), fall back to salvaging a JSON object or a plausible
    final value from inside the trace.
    """
    if not text:
        return ""

    think_match = re.search(r"<think>([\s\S]*?)</think>\s*([\s\S]*)", text)
    if not think_match:
        return text.strip()

    after = think_match.group(2).strip()
    if after:
        return after

    inside = think_match.group(1).strip()

    json_match = re.search(r'\{[^{}]*"result"[^{}]*\}', inside)
    if json_match:
        return json_match.group(0)

    json_match = re.search(r'(\{[\s\S]*\})\s*$', inside)
    if json_match:
        return json_match.group(1)

    for line in reversed(inside.splitlines()):
        line = line.strip()
        if not line or line.endswith("...") or line.endswith(":"):
            continue
        if line[0] in '{["\'0123456789-' or line.lower() in ("true", "false", "none", "null"):
            return line

    return inside
