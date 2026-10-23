"""Backend registry and factory."""

from __future__ import annotations

from typing import Dict

from .base import Backend, GenerateResponse, ToolUseDetected, detect_tool_use
from .reasoning import ReasoningCapabilities
from .ollama import OllamaBackend
from .hf import HFBackend
from .openai_compat import OpenAICompatBackend

__all__ = [
    "Backend",
    "GenerateResponse",
    "ReasoningCapabilities",
    "ToolUseDetected",
    "detect_tool_use",
    "OllamaBackend",
    "HFBackend",
    "OpenAICompatBackend",
    "create_backend",
]

_REGISTRY: Dict[str, type] = {
    "ollama": OllamaBackend,
    "hf": HFBackend,
    "openai": OpenAICompatBackend,
}


def create_backend(name: str, **kwargs) -> Backend:
    """
    Create a backend by name.

    >>> backend = create_backend("ollama", model="qwen3:4b")
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {list(_REGISTRY.keys())}"
        )
    return cls(**kwargs)
