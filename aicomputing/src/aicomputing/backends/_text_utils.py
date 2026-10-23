"""Visible thinking-block cleanup shared by inference backends.

This removes recognized leading text blocks, not a model's hidden computation.
It does not enable, disable or verify native reasoning.
"""

from __future__ import annotations


def strip_thinking(text: str) -> str:
    """Remove leading <think>...</think> blocks from response text.

    Return empty text if a leading block is unclosed or no final text remains.
    Preserve occurrences of the tags within the remaining response text.
    """
    if not text:
        return ""

    text = text.strip()
    while text.startswith("<think>"):
        closing = text.find("</think>", len("<think>"))
        if closing == -1:
            return ""
        text = text[closing + len("</think>"):].lstrip()
    return text
