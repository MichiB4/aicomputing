"""Remove visible thinking blocks without controlling native reasoning."""

from __future__ import annotations


def strip_thinking(text: str) -> str:
    """Remove leading <think> blocks; an unclosed block yields empty text.

    Tags within the remaining response are preserved.
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
