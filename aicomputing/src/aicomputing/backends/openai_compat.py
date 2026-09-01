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

from ..model_config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_THINK,
    ThinkingMode,
    get_thinking_mode,
    resolve_thinking,
)
from ._text_utils import append_no_think, strip_thinking
from .base import (
    Backend,
    GenerateResponse,
    ToolUseDetected,
    detect_tool_use,
    normalize_finish_reason,
)

logger = logging.getLogger("aicomputing")


class OpenAICompatBackend(Backend):
    """Backend using an OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        allow_tool_use: bool = False,
    ) -> None:
        self._model_name = model
        self._base_url = base_url
        self._api_key = api_key
        self._client = None
        # Server-side tool use invalidates spec-only execution reasoning: an
        # interpreter, not the model, would have produced the value. Requests
        # never declare tools, so this only fires on a provider-side default.
        # Leave it False for anything whose numbers reach the thesis.
        self._allow_tool_use = allow_tool_use

    # Retries for transient HTTP failures, chiefly 429. The SDK default is 2
    # with a sub-second backoff, which is tuned for a request-rate limit and
    # is too short for a TOKEN-rate one: a free-tier tokens-per-minute bucket
    # can need most of a minute to refill, and the reasoning arms spend
    # thousands of tokens per call.
    #
    # Observed on Groq gpt-oss-120b (8000 TPM): "Used 5271, Requested 2763,
    # try again in 255ms" aborted a benchmark cell 312 seconds and ~20
    # completed examples in. The SDK honours Retry-After, so the fix is to let
    # it wait rather than to hand-roll a backoff here.
    _MAX_RETRIES = 6

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs: Dict[str, Any] = {"max_retries": self._MAX_RETRIES}
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
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        stop: Optional[List[str]] = None,
        think: bool = DEFAULT_THINK,
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
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        stop: Optional[List[str]] = None,
        think: bool = DEFAULT_THINK,
        **kwargs,
    ) -> GenerateResponse:
        client = self._get_client()
        mdl = model or self._model_name
        thinking_mode = get_thinking_mode(mdl)

        # Single source of truth for what this model will actually do with the
        # requested mode. The backend applies the resolution; it does not
        # decide it, and no strategy branches on model name.
        resolution = resolve_thinking(mdl, think)

        # --- Resolve thinking behaviour per model family ---
        if thinking_mode == ThinkingMode.ALWAYS_THINKS:
            if not think:
                logger.warning(
                    "Model '%s' always generates reasoning traces — "
                    "`think=False` has no effect. The result records "
                    "effective_think=True and thinking_honoured=False so this "
                    "call is not mistaken for a non-thinking condition.",
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
        raw_dump = response.model_dump() if hasattr(response, "model_dump") else None

        # Validity guard, not error handling — see ToolUseDetected in base.py.
        # This must run before the content is read: a tool call returns
        # content=None, which would otherwise become "" and be indistinguishable
        # from an ordinary parse failure further downstream.
        if not self._allow_tool_use:
            evidence = detect_tool_use(choice, raw_dump)
            if evidence is not None:
                raise ToolUseDetected(
                    f"Provider ran a tool for model {mdl!r} ({evidence}). This package "
                    f"never declares tools, so a provider-side default is in play. The "
                    f"returned value would measure an interpreter rather than the "
                    f"model's reasoning, so it is refused. Disable server-side tools "
                    f"for this model, or construct the backend with allow_tool_use=True "
                    f"if you deliberately want tool-assisted numbers — they are not "
                    f"spec-only execution reasoning and must not be reported as such."
                )

        raw_text = (choice.message.content or "").strip()
        tokens = input_tokens = output_tokens = None
        if response.usage:
            tokens = response.usage.completion_tokens
            # getattr rather than attribute access: the usage object varies
            # across OpenAI-compatible providers and a missing field must read
            # as "not reported", not raise.
            input_tokens = getattr(response.usage, "prompt_tokens", None)
            output_tokens = getattr(response.usage, "completion_tokens", None)
        # Map the provider's spelling onto {"stop", "length"}; the strategy
        # layer tests for "length" exactly when deciding to retry. The
        # un-normalised value is carried separately — the mapping is lossy and
        # an unrecognised value is exactly the audit signal worth keeping.
        provider_finish_reason = getattr(choice, "finish_reason", None)
        finish_reason = normalize_finish_reason(provider_finish_reason, mdl)

        raw_text = strip_thinking(raw_text)

        return GenerateResponse(
            text=raw_text,
            model=mdl,
            tokens_used=tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            requested_think=resolution.requested,
            effective_think=resolution.effective,
            thinking_honoured=resolution.honoured,
            finish_reason=finish_reason,
            provider_finish_reason=(
                str(provider_finish_reason) if provider_finish_reason is not None else None
            ),
            raw=raw_dump,
        )

    def is_available(self) -> bool:
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False
