"""OpenAI-compatible chat backend with configurable endpoint and model.

None omits the token cap; positive allowances use max_tokens. Provider limits apply.
Optional reasoning_effort is sent literally; endpoint/model support determines
its effect. No model-name inference or reasoning directives are added.
Returned text has leading thinking blocks stripped.

Install: pip install aicomputing[openai]
"""

from __future__ import annotations

import os
import math
import time
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any, Dict, List, Optional

from ..model_config import DEFAULT_MAX_NEW_TOKENS
from ..request import validate_generation_limits
from ._text_utils import strip_thinking
from .diagnostics import exception_diagnostics, safe_request_id
from .base import (
    Backend,
    BackendCommunicationError,
    BackendTimeoutError,
    GenerateResponse,
    ToolUseDetected,
    detect_tool_use,
    normalize_finish_reason,
)


class OpenAICompatBackend(Backend):
    """Backend using an OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        allow_tool_use: bool = False,
        *,
        reasoning_effort: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        sdk_max_retries: Optional[int] = None,
        request_diagnostics: bool = False,
    ) -> None:
        if reasoning_effort is not None:
            if not isinstance(reasoning_effort, str):
                raise TypeError("reasoning_effort must be a string or None")
            if not reasoning_effort.strip():
                raise ValueError("reasoning_effort must not be empty")

        if timeout_seconds is not None and (
            type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive, or None")
        if sdk_max_retries is not None and (type(sdk_max_retries) is not int or sdk_max_retries < 0):
            raise ValueError("sdk_max_retries must be a nonnegative integer or None")
        if type(request_diagnostics) is not bool:
            raise TypeError("request_diagnostics must be a boolean")
        self._timeout_seconds = timeout_seconds
        self._sdk_max_retries = sdk_max_retries
        self._request_diagnostics = request_diagnostics
        self._reasoning_effort = reasoning_effort
        self._model_name = model
        self._base_url = base_url
        self._api_key = api_key
        self._client = None
        # Bypasses the response guard; does not enable or execute tools.
        self._allow_tool_use = allow_tool_use

    reasoning_parameter = "reasoning_effort"
    _reasoning_value_types = (str,)

    def _reasoning_endpoint(self) -> str:
        if self._client is not None:
            endpoint = getattr(self._client, "base_url", None)
            if endpoint is None:
                raise ValueError("Cannot resolve this OpenAI-compatible client's endpoint")
            return str(endpoint).rstrip("/")
        endpoint = self._base_url or os.environ.get("OPENAI_BASE_URL")
        import httpx
        endpoint = "https://api.openai.com/v1" if endpoint is None else endpoint
        return str(httpx.URL(endpoint)).rstrip("/")

    # SDK retries are separate from strategy and validation retries.
    _MAX_RETRIES = 2

    def _get_client(self):
        """Create the SDK client on first use and reuse it afterward."""
        if self._client is None:
            from openai import OpenAI

            kwargs: Dict[str, Any] = {"max_retries": self._MAX_RETRIES}
            if self._sdk_max_retries is not None:
                kwargs["max_retries"] = self._sdk_max_retries
            if self._timeout_seconds is not None:
                kwargs["timeout"] = self._timeout_seconds
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = OpenAI(**kwargs)
        return self._client

    def close(self) -> None:
        """Close an initialized client; later generation recreates it.

        On failure, retain the client for another cleanup attempt.
        Do not close during generation.
        """
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return self._model_name

    def cache_state(self) -> Optional[Dict[str, Any]]:
        """Describe this adapter's configuration without exposing credentials."""
        if type(self) is not OpenAICompatBackend:
            # Subclasses must explicitly account for their additional behavior.
            return None
        return {
            "base_url": self._base_url,
            "default_model": self._model_name,
            "reasoning_effort": self._reasoning_effort,
            "allow_tool_use": self._allow_tool_use,
            "timeout_seconds": self._timeout_seconds,
            "sdk_max_retries": self._sdk_max_retries,
            "request_diagnostics": self._request_diagnostics,
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
        **kwargs,
    ) -> GenerateResponse:
        return self.generate_chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
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
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> GenerateResponse:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported generation options: {names}")
        validate_generation_limits(max_new_tokens, 0)

        client = self._get_client()
        mdl = model or self._model_name

        api_kwargs: Dict[str, Any] = {}
        if max_new_tokens is not None:
            api_kwargs["max_tokens"] = max_new_tokens
        if stop:
            api_kwargs["stop"] = list(stop)
        if self._reasoning_effort is not None:
            api_kwargs["reasoning_effort"] = self._reasoning_effort

        # Keep this optional SDK import local to the backend operation.
        from openai import APIError, APITimeoutError

        diagnostics = None
        if self._request_diagnostics:
            client_id = str(uuid4())
            api_kwargs["extra_headers"] = {"X-Client-Request-Id": client_id}
            diagnostics = {
                "client_request_id": client_id,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "http_phase_timeout_seconds": self._timeout_seconds,
                "timeout_is_total_request_deadline": False,
                "sdk_max_retries": self._MAX_RETRIES if self._sdk_max_retries is None else self._sdk_max_retries,
            }
        started = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=mdl,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                **api_kwargs,
            )
        except APITimeoutError as exc:
            if diagnostics is not None:
                diagnostics.update(elapsed_seconds=time.monotonic() - started, **exception_diagnostics(exc))
            raise BackendTimeoutError(f"{type(exc).__name__}: {exc}", diagnostics=diagnostics) from exc
        except APIError as exc:
            if diagnostics is not None:
                diagnostics.update(elapsed_seconds=time.monotonic() - started, **exception_diagnostics(exc))
            raise BackendCommunicationError(f"{type(exc).__name__}: {exc}", diagnostics=diagnostics) from exc

        if diagnostics is not None:
            diagnostics.update(elapsed_seconds=time.monotonic() - started,
                               provider_request_id=safe_request_id(getattr(response, "_request_id", None)))

        choice = response.choices[0]
        raw_dump = response.model_dump() if hasattr(response, "model_dump") else None

        # Reject tool indicators before missing text becomes a parse failure.
        if not self._allow_tool_use:
            evidence = detect_tool_use(choice, raw_dump)
            if evidence is not None:
                raise ToolUseDetected(
                    f"Response for model {mdl!r} contains tool-use indicators "
                    f"({evidence}) and cannot be accepted as an unaided prediction. "
                    "Set allow_tool_use=True to bypass this check. "
                    "Bypassing the check does not execute tools."
                )

        raw_text = (choice.message.content or "").strip()
        input_tokens = output_tokens = None
        usage = getattr(response, "usage", None)
        if usage is not None:
            # Missing usage fields remain unknown rather than being estimated.
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)

        provider_finish_reason = getattr(choice, "finish_reason", None)
        finish_reason = normalize_finish_reason(provider_finish_reason, mdl)

        raw_text = strip_thinking(raw_text)

        return GenerateResponse(
            text=raw_text,
            model=mdl,
            tokens_used=output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            requested_reasoning=(
                {"reasoning_effort": api_kwargs["reasoning_effort"]}
                if "reasoning_effort" in api_kwargs else {}
            ),
            finish_reason=finish_reason,
            provider_finish_reason=(
                str(provider_finish_reason) if provider_finish_reason is not None else None
            ),
            raw=raw_dump,
            transport_diagnostics=diagnostics,
        )

    def is_available(self) -> bool:
        """Check SDK importability without contacting the provider."""
        try:
            import openai  # noqa: F401

            return True
        except ImportError:
            return False
