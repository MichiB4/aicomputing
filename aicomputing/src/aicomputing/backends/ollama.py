"""
Ollama backend for AI Computing.

Talks to a local Ollama server. Supports a `think` parameter that
strategies use to control whether the model reasons internally.

Token budgets are determined by the caller (via model_config.get_token_budget)
and passed in as max_new_tokens. This backend uses them exactly as given —
no floors, no multipliers. Retry-on-cutoff is handled by the strategy layer.

finish_reason is "stop" for a clean completion, "length" if the model hit
the token limit before finishing.

Compatible with ollama Python package >= 0.4 (Pydantic response objects).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..model_config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_THINK,
    ThinkingMode,
    get_thinking_mode,
)
from ._text_utils import append_no_think, strip_thinking
from .base import Backend, GenerateResponse, normalize_finish_reason


class OllamaBackend(Backend):
    """Backend using a local Ollama server."""

    def __init__(
        self,
        model: str = "qwen3:4b",
        host: Optional[str] = None,
    ) -> None:
        self._default_model = model
        self._host = host
        self._client = None

    def _get_client(self):
        if self._client is None:
            import ollama as _ollama
            if self._host:
                self._client = _ollama.Client(host=self._host)
            else:
                self._client = _ollama
        return self._client

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def default_model(self) -> str:
        return self._default_model

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        think: bool = DEFAULT_THINK,
        **kwargs,
    ) -> GenerateResponse:
        # Qwen3 ignores /no_think on the generate() endpoint but respects
        # it on chat(). Route all calls through generate_chat for consistency.
        messages = [{"role": "user", "content": prompt}]
        return self.generate_chat(
            messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
            think=think,
            stop=stop,
        )

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        seed: Optional[int] = None,
        think: bool = DEFAULT_THINK,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> GenerateResponse:
        client = self._get_client()
        mdl = model or self._default_model

        # Only toggle-capable models (Qwen3 etc.) understand /no_think.
        # Appending it to Llama/Phi/... prompts just adds noise.
        if not think and get_thinking_mode(mdl) == ThinkingMode.THINKING_TOGGLE:
            chat_messages = append_no_think(messages)
        else:
            chat_messages = messages

        options: Dict[str, Any] = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "num_predict": int(max_new_tokens),
        }
        if seed is not None:
            options["seed"] = int(seed)
        if stop:
            options["stop"] = list(stop)

        response = client.chat(model=mdl, messages=chat_messages, options=options)

        msg = _get_field(response, "message", {})
        raw_text = _get_field(msg, "content", "")
        tokens = _get_field(response, "eval_count", None)
        # Shared normalisation so every backend maps onto {"stop", "length"}
        # identically and unknown values are surfaced rather than swallowed.
        provider_finish_reason = _get_field(response, "done_reason", None)
        finish_reason = normalize_finish_reason(provider_finish_reason, mdl)
        raw_text = strip_thinking(raw_text)

        return GenerateResponse(
            text=raw_text,
            model=mdl,
            tokens_used=tokens,
            finish_reason=finish_reason,
            provider_finish_reason=(
                str(provider_finish_reason) if provider_finish_reason is not None else None
            ),
            raw=_to_dict(response),
        )

    def is_available(self) -> bool:
        try:
            client = self._get_client()
            client.list()
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_field(obj: Any, field: str, default: Any = None) -> Any:
    """Get a field from a Pydantic object or dict."""
    val = getattr(obj, field, None)
    if val is not None:
        return val
    if isinstance(obj, dict):
        return obj.get(field, default)
    return default


def _to_dict(obj: Any) -> Any:
    """Convert Pydantic model to dict, or return as-is."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return vars(obj) if hasattr(obj, "__dict__") else obj
