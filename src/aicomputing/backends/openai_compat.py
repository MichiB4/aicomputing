"""
OpenAI-compatible API backend for AI Computing.

Works with any server that exposes the OpenAI chat completions API:
Groq, Cerebras, OpenRouter, SambaNova, vLLM, LM Studio, etc.

Thinking behaviour
------------------
Different model families handle the `think` flag differently:

  ALWAYS_THINKS (DeepSeek-R1)
    The model always reasons. `think=False` is logged as a warning but
    the request is sent unchanged — the thinking trace will be stripped
    from the visible output by _strip_thinking().

  THINKING_TOGGLE (Qwen3, GPT-OSS-120B)
    When `think=False`, the string ``/no_think`` is appended to the last
    user message. This is the same convention used by the Ollama backend
    and matches Groq/Cerebras's documented Qwen3 toggle mechanism.

  NO_THINKING (Llama, Phi, Mistral, etc.)
    The `think` parameter has no effect — these models don't generate
    reasoning traces regardless.

Install: pip install aicomputing[openai]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..model_config import ThinkingMode, get_thinking_mode
from ._text_utils import append_no_think, strip_thinking
from .base import Backend, GenerateResponse

logger = logging.getLogger("aicomputing")


class OpenAICompatBackend(Backend):
    """Backend using an OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._model_name = model
        self._base_url = base_url
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs: Dict[str, Any] = {}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = OpenAI(**kwargs)
        return self._client

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 256,
        stop: Optional[List[str]] = None,
        think: bool = True,
        **kwargs,
    ) -> GenerateResponse:
        return self.generate_chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            stop=stop,
            think=think,
            **kwargs,
        )

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 256,
        stop: Optional[List[str]] = None,
        think: bool = True,
        **kwargs,
    ) -> GenerateResponse:
        client = self._get_client()
        mdl = model or self._model_name
        thinking_mode = get_thinking_mode(mdl)

        # --- Resolve thinking behaviour per model family ---
        if thinking_mode == ThinkingMode.ALWAYS_THINKS:
            if not think:
                logger.warning(
                    "Model '%s' always generates reasoning traces — "
                    "`think=False` has no effect. "
                    "Use a THINKING_TOGGLE model (e.g. Qwen3) to disable thinking.",
                    mdl,
                )
            effective_messages = messages

        elif thinking_mode == ThinkingMode.THINKING_TOGGLE:
            if think:
                effective_messages = messages
            else:
                effective_messages = append_no_think(messages)

        else:
            # NO_THINKING — model never reasons, flag is irrelevant
            effective_messages = messages

        api_kwargs: Dict[str, Any] = {}
        if stop:
            api_kwargs["stop"] = list(stop)

        response = client.chat.completions.create(
            model=mdl,
            messages=effective_messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            **api_kwargs,
        )

        choice = response.choices[0]
        raw_text = (choice.message.content or "").strip()
        tokens = None
        if response.usage:
            tokens = response.usage.completion_tokens
        finish_reason = getattr(choice, "finish_reason", None) or "stop"

        raw_text = strip_thinking(raw_text)

        return GenerateResponse(
            text=raw_text,
            model=mdl,
            tokens_used=tokens,
            finish_reason=finish_reason,
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    def is_available(self) -> bool:
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False
