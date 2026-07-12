"""
Abstract backend interface for AI Computing.

A backend handles the raw communication with an LLM: sending a prompt
and receiving generated text. Different backends talk to different
inference engines (Ollama, HuggingFace, OpenAI API, etc.) but expose
the same interface to the strategy layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class GenerateResponse:
    """Raw response from a backend."""
    text: str
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None   # "stop" | "length" | None
    raw: Optional[Dict[str, Any]] = None  # backend-specific metadata


class Backend(ABC):
    """Abstract base for all inference backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'ollama', 'hf', 'openai'."""

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model name when none is specified."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 512,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> GenerateResponse:
        """Send a prompt and return the generated text."""

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 512,
        **kwargs,
    ) -> GenerateResponse:
        """
        Chat-style generation (system + user messages).
        Default implementation concatenates into a single prompt.
        """
        prompt = "\n\n".join(msg.get("content", "") for msg in messages)
        return self.generate(
            prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    def is_available(self) -> bool:
        """Check if the backend is reachable / importable."""
        return True
