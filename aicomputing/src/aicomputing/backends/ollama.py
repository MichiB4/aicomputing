"""Ollama chat backend with optional native think control.

Omitting think uses the model default; SDK, server and model determine support.
Recorded settings describe requests, not proof of their effect.

None sends num_predict=-1; positive token allowances pass through unchanged.
The strategy layer rejects truncated replies using normalized finish reasons.
Replies with tool-call indicators are rejected before returning model text.

Install: pip install aicomputing[ollama]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..request import validate_generation_limits
from ..validation import validate_nonblank_text
from ._text_utils import strip_thinking
from .base import (
    Backend,
    BackendCommunicationError,
    BackendTimeoutError,
    GenerateResponse,
    ToolUseDetected,
    detect_tool_use,
    normalize_finish_reason,
)


class OllamaBackend(Backend):
    """Backend using a local Ollama server."""

    def __init__(
        self,
        model: str = "qwen3:4b",
        host: Optional[str] = None,
        *,
        think: bool | str | None = None,
    ) -> None:
        validate_nonblank_text(model, "model")
        validate_nonblank_text(host, "host", optional=True)
        if think is not None:
            if not isinstance(think, (bool, str)):
                raise TypeError("think must be a bool, a string, or None")
            if isinstance(think, str) and not think.strip():
                raise ValueError("think must not be an empty string")

        self._think = think
        self._default_model = model
        self._host = host
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from ollama import Client
            except ModuleNotFoundError as exc:
                if exc.name != "ollama":
                    raise
                raise ImportError(
                    "Ollama backend requires the optional Ollama SDK; "
                    "install aicomputing[ollama] to enable it."
                ) from exc
            self._client = Client(host=self._host)
        return self._client

    def close(self) -> None:
        """Close this backend's transport, retaining it if cleanup fails.

        Cleanup is lazy and repeatable. Later generation reacquires a client.
        Do not close during generation.
        """
        if self._client is not None:
            # Ollama's synchronous Client stores its HTTPX transport here;
            # the SDK does not expose a public close method.
            self._client._client.close()
        self._client = None

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def default_model(self) -> str:
        return self._default_model

    def cache_state(self) -> Optional[Dict[str, Any]]:
        """Describe this adapter's endpoint, model and native thinking request."""
        if type(self) is not OllamaBackend:
            # Subclasses must explicitly account for their additional behavior.
            return None
        return {
            "host": self._host,
            "default_model": self._default_model,
            "think": self._think,
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> GenerateResponse:
        # Use the same chat request path for prompt and message inputs.
        messages = [{"role": "user", "content": prompt}]
        return self.generate_chat(
            messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
            stop=stop,
            **kwargs,
        )

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> GenerateResponse:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported generation options: {names}")
        validate_nonblank_text(model, "model", optional=True)
        validate_generation_limits(max_new_tokens, 0, temperature=temperature, top_p=top_p)

        client = self._get_client()
        mdl = self._default_model if model is None else model

        request_kwargs: Dict[str, Any] = {}
        if self._think is not None:
            request_kwargs["think"] = self._think

        options: Dict[str, Any] = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            # Ollama documents -1 as unlimited generation. Send it explicitly
            # so a finite Modelfile default does not reinstate an output cap.
            "num_predict": -1 if max_new_tokens is None else max_new_tokens,
        }
        if seed is not None:
            options["seed"] = int(seed)
        if stop:
            options["stop"] = list(stop)

        # These dependencies are needed only when the Ollama backend is used.
        from httpx import (
            HTTPError,
            LocalProtocolError,
            TimeoutException,
            UnsupportedProtocol,
        )
        from ollama import ResponseError

        try:
            response = client.chat(
                model=mdl, messages=messages, options=options, **request_kwargs,
            )
        except (LocalProtocolError, UnsupportedProtocol):
            # Invalid client request configuration is a local error.
            raise
        except (TimeoutException, TimeoutError) as exc:
            raise BackendTimeoutError(f"{type(exc).__name__}: {exc}") from exc
        except (ResponseError, HTTPError, ConnectionError) as exc:
            raise BackendCommunicationError(f"{type(exc).__name__}: {exc}") from exc

        raw_dump = _to_dict(response)
        provider_finish_reason = _get_field(response, "done_reason", None)
        evidence = detect_tool_use(response, raw_dump, finish_reason=provider_finish_reason)
        if evidence is not None:
            raise ToolUseDetected(
                f"Response for model {mdl!r} contains tool-use indicators "
                f"({evidence}) and cannot be accepted as an unaided prediction."
            )

        msg = _get_field(response, "message", {})
        raw_text = _get_field(msg, "content", "")
        tokens = _get_field(response, "eval_count", None)
        finish_reason = normalize_finish_reason(provider_finish_reason, mdl)
        thinking = _get_field(msg, "thinking", None)
        observed_reasoning = {
            "thinking_field_type": type(thinking).__name__,
            "thinking_chars": len(thinking) if isinstance(thinking, str) else None,
            "content_has_think_open": "<think>" in (raw_text or ""),
            "content_has_think_close": "</think>" in (raw_text or ""),
        }
        return GenerateResponse(
            text=raw_text,
            model=mdl,
            text_for_parsing=strip_thinking(raw_text),
            input_tokens=_get_field(response, "prompt_eval_count", None),
            output_tokens=tokens,
            requested_reasoning=dict(request_kwargs),
            observed_reasoning=observed_reasoning,
            finish_reason=finish_reason,
            provider_finish_reason=(
                str(provider_finish_reason) if provider_finish_reason is not None else None
            ),
            raw=raw_dump,
        )



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
