"""Ollama chat backend with optional native think control.

Omitting think uses the model default; SDK, server and model determine support.
Recorded settings describe requests, not proof of their effect.

None sends num_predict=-1; positive token allowances pass through unchanged.
The strategy layer handles truncation retries using normalized finish reasons.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from ..model_config import DEFAULT_MAX_NEW_TOKENS
from ..request import validate_generation_limits
from ._text_utils import strip_thinking
from .base import (
    Backend,
    BackendCommunicationError,
    BackendTimeoutError,
    GenerateResponse,
    ReasoningCapabilities,
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
        if think is not None:
            if not isinstance(think, (bool, str)):
                raise TypeError("think must be a bool, a string, or None")
            if isinstance(think, str) and not think.strip():
                raise ValueError("think must not be an empty string")

        self._think = think
        self._default_model = model
        self._host = host
        self._client = None
        self._owns_client = False

    reasoning_parameter = "think"
    _reasoning_value_types = (bool, str)

    def _reasoning_transport(self):
        sdk = self._client
        if sdk is None and not self._host:
            sdk = sys.modules.get("ollama")
        if sdk is None:
            return None
        owner = getattr(getattr(sdk, "chat", None), "__self__", None)
        transport = getattr(owner, "_client", None)
        if transport is None or getattr(transport, "base_url", None) is None:
            raise ValueError("Cannot resolve this Ollama client's metadata transport")
        return transport

    def _reasoning_endpoint(self) -> str:
        transport = self._reasoning_transport()
        if transport is not None:
            return str(transport.base_url).rstrip("/")
        # Match SDK host defaults, including bare hosts and URL path prefixes.
        host = self._host or os.environ.get("OLLAMA_HOST") or ""
        scheme, separator, address = host.partition("://")
        if not separator:
            scheme, address, port = "http", host, 11434
        else:
            port = {"http": 80, "https": 443}.get(scheme, 11434)
        if scheme.lower() not in ("http", "https"):
            raise ValueError("Metadata discovery requires HTTP or HTTPS")
        parts = urlsplit(f"{scheme}://{address}")
        hostname = parts.hostname or "127.0.0.1"
        if ":" in hostname:
            hostname = f"[{hostname}]"
        path = parts.path.strip("/")
        suffix = f"/{path}" if path else ""
        import httpx
        return str(httpx.URL(f"{scheme}://{hostname}:{parts.port or port}{suffix}")).rstrip("/")

    def reasoning_capabilities(
        self, model: Optional[str] = None, *, discover: bool = False,
    ) -> ReasoningCapabilities:
        local = super().reasoning_capabilities(model, discover=discover)
        if not discover:
            return local
        import httpx

        # Reuse the generation client's endpoint, headers and authentication.
        # SDK show() would discard fields unknown to its response model.
        self._get_client()
        transport = self._reasoning_transport()
        local = super().reasoning_capabilities(model, discover=False)
        try:
            response = transport.post("/api/show", json={"model": local.model}, timeout=10.0)
            response.raise_for_status()
            payload = response.json()
        except (httpx.LocalProtocolError, httpx.UnsupportedProtocol):
            raise
        except httpx.TimeoutException as exc:
            raise BackendTimeoutError(f"{type(exc).__name__}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise BackendCommunicationError(f"{type(exc).__name__}: {exc}") from exc
        except ValueError as exc:
            raise BackendCommunicationError("Invalid Ollama metadata JSON") from exc

        try:
            if not isinstance(payload, dict):
                raise ValueError("Metadata must be an object")
            if "thinking" not in payload:
                return ReasoningCapabilities(
                    backend=local.backend, model=local.model, parameter=local.parameter,
                    endpoint=local.endpoint, source="ollama_api_show",
                )
            thinking = payload["thinking"]
            if not isinstance(thinking, dict) or not isinstance(thinking.get("values"), list):
                raise ValueError("thinking.values must be a list")
            return ReasoningCapabilities(
                backend=local.backend, model=local.model, parameter=local.parameter,
                endpoint=local.endpoint, support="supported",
                values=thinking["values"], default=thinking.get("default"),
                source="ollama_api_show",
            )
        except (TypeError, ValueError) as exc:
            raise BackendCommunicationError("Invalid advertised thinking controls") from exc

    def _get_client(self):
        if self._client is None:
            import ollama as _ollama
            if self._host:
                self._client = _ollama.Client(host=self._host)
                self._owns_client = True
            else:
                self._client = _ollama
        return self._client

    def close(self) -> None:
        """Close owned transport; shared SDK clients stay open.

        Cleanup is lazy and repeatable. Later generation reacquires a client.
        Do not close during generation.
        """
        if self._owns_client:
            # Ollama's synchronous Client stores its HTTPX transport here;
            # the SDK does not expose a public close method.
            self._client._client.close()
        self._client = None
        self._owns_client = False

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
        max_new_tokens: Optional[int] = DEFAULT_MAX_NEW_TOKENS,
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
        max_new_tokens: Optional[int] = DEFAULT_MAX_NEW_TOKENS,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> GenerateResponse:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported generation options: {names}")
        validate_generation_limits(max_new_tokens, 0)

        client = self._get_client()
        mdl = model or self._default_model

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

        msg = _get_field(response, "message", {})
        raw_text = _get_field(msg, "content", "")
        tokens = _get_field(response, "eval_count", None)
        provider_finish_reason = _get_field(response, "done_reason", None)
        finish_reason = normalize_finish_reason(provider_finish_reason, mdl)
        thinking = _get_field(msg, "thinking", None)
        observed_reasoning = {
            "thinking_field_type": type(thinking).__name__,
            "thinking_chars": len(thinking) if isinstance(thinking, str) else None,
            "content_has_think_open": "<think>" in (raw_text or ""),
            "content_has_think_close": "</think>" in (raw_text or ""),
        }
        raw_text = strip_thinking(raw_text)

        return GenerateResponse(
            text=raw_text,
            model=mdl,
            tokens_used=tokens,
            input_tokens=_get_field(response, "prompt_eval_count", None),
            output_tokens=tokens,
            requested_reasoning=dict(request_kwargs),
            observed_reasoning=observed_reasoning,
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
