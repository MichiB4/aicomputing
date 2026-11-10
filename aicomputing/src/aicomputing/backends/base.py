"""Shared inference interface, response metadata, and tool-use guards."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..request import validate_generation_limits
from ..validation import validate_nonblank_text

logger = logging.getLogger("aicomputing")


def _close_after_error(close: Callable[[], None]) -> None:
    """Attempt cleanup without replacing an active error with a cleanup error."""
    try:
        close()
    except Exception as exc:
        logger.warning("Backend cleanup also failed with %s; preserving the original error.",
                       type(exc).__name__)


class BackendCommunicationError(RuntimeError):
    """A provider API or transport failed; the original error is the cause."""

    def __init__(self, *args, diagnostics=None):
        super().__init__(*args)
        self.diagnostics = diagnostics


class BackendTimeoutError(BackendCommunicationError):
    """A provider API request timed out."""


class ToolUseDetected(RuntimeError):
    """A response was rejected for tool-use indicators, not a parsing failure.

    Indicators can describe requested calls; they do not prove tool execution.
    """


@dataclass
class GenerateResponse:
    """Original assistant text, optional parsing view, and provider evidence."""
    text: str
    model: str

    # Cleanup for parsing must not replace the original assistant text.
    # None uses text unchanged; an empty string is an intentionally empty view.
    text_for_parsing: Optional[str] = None

    # Provider-reported usage; None means unreported, never estimated.
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    # Native reasoning overrides sent for this call, not proof of their effect.
    # {} means no override was sent; None means the backend did not record it.
    requested_reasoning: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None   # "stop" | "length" | None
    raw: Optional[Dict[str, Any]] = None  # backend-specific metadata

    # Original reason retained because normalization collapses unknowns to "stop".
    provider_finish_reason: Optional[str] = None

    # Response measurements before text cleanup, not an inferred native mode.
    # None means this backend did not record observations.
    observed_reasoning: Optional[Dict[str, Any]] = None

    # Optional per-SDK-call transport evidence, never a response body or URL.
    transport_diagnostics: Optional[Dict[str, Any]] = None

    @property
    def parsing_text(self) -> str:
        return self.text if self.text_for_parsing is None else self.text_for_parsing


# Normalize aliases once; strategies recognize truncation as exactly "length".
_LENGTH_ALIASES = {alias.replace("_", "") for alias in (
    "length", "max_tokens", "maxtokens", "max_output_tokens",
    "model_length", "token_limit", "truncated",
)}

# Non-truncation reasons, including filters and tool calls, normalize to stop.
_STOP_ALIASES = {alias.replace("_", "") for alias in (
    "stop", "end_turn", "eos", "eos_token", "stop_sequence",
    "complete", "completed", "finished", "content_filter",
    "tool_calls", "tool_use", "function_call",
)}


# Tool-use field names; prose values are ignored.
_TOOL_USE_KEYS = frozenset({
    "tool_calls", "tool_call", "function_call", "tool_use",
    "code_execution", "code_execution_result", "executable_code",
    "code_interpreter", "python_call", "tool_outputs", "tool_result",
})

# finish_reason values that mean "I stopped to run a tool".
_TOOL_USE_FINISH_REASONS = frozenset({
    "tool_calls", "tool_call", "function_call", "tool_use",
})


def _iter_keys(obj: Any):
    """Yield every dict key in a nested structure, ignoring all values."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key)
            yield from _iter_keys(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_keys(value)


def detect_tool_use(
    choice: Any, raw_dump: Optional[Dict[str, Any]] = None,
    *, finish_reason: Optional[str] = None,
) -> Optional[str]:
    """Describe tool-use indicators, or return None.

    Check message fields, finish reasons, and populated raw-response keys.
    Prose values are ignored to avoid rejecting answers that discuss tools.
    Supply finish_reason when the provider uses a different termination field.
    """
    message = getattr(choice, "message", None)
    if message is not None:
        if getattr(message, "tool_calls", None):
            return "message.tool_calls is populated"
        if getattr(message, "function_call", None):
            return "message.function_call is populated"

    if finish_reason is None:
        finish_reason = getattr(choice, "finish_reason", None)
    finish = str(finish_reason or "").strip().lower()
    if finish.replace("-", "_") in _TOOL_USE_FINISH_REASONS:
        return f"finish_reason={finish!r}"

    if raw_dump:
        for key in _iter_keys(raw_dump):
            normalised = key.strip().lower().replace("-", "_")
            if normalised in _TOOL_USE_KEYS:
                # SDKs may serialize unused fields as null.
                if _key_is_populated(raw_dump, key):
                    return f"response contains a populated {key!r} field"
    return None


def _key_is_populated(obj: Any, target: str) -> bool:
    """True if `target` appears anywhere in `obj` with a truthy value."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) == target and value:
                return True
            if _key_is_populated(value, target):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_key_is_populated(v, target) for v in obj)
    return False


def normalize_finish_reason(value: Optional[str], model: str = "") -> str:
    """Map finish reasons to "stop" or "length", ignoring case and separators.

    Unknowns warn and default to "stop"; the original value remains recorded.
    """
    if value is None:
        return "stop"

    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    # Collapse camelCase ("maxTokens" -> "maxtokens") for the alias lookup.
    flat = key.replace("_", "")

    if flat in _LENGTH_ALIASES:
        return "length"
    if flat in _STOP_ALIASES:
        return "stop"

    logger.warning(
        "Unrecognised finish_reason %r from model %r — treating as 'stop'. "
        "Add the value to _LENGTH_ALIASES or _STOP_ALIASES in backends/base.py if it recurs.",
        value, model or "<unknown>",
    )
    return "stop"


class Backend(ABC):
    """Abstract base for all inference backends."""

    def close(self) -> None:
        """Release owned resources; the default is a no-op.

        Overrides must remain lazy and tolerate repeated calls.
        Do not close during generation.
        """

    def __enter__(self) -> Backend:
        """Return this backend without loading clients or models."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Release resources, preserving an active error if cleanup also fails."""
        if exc_type is None:
            self.close()
        else:
            _close_after_error(self.close)

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'ollama', 'hf', 'openai'."""

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model name when none is specified."""

    def cache_state(self) -> Optional[Dict[str, Any]]:
        """Return non-secret generation/acceptance settings, or None to bypass caching.

        The runtime adds class identity and request generation settings.
        """
        return None

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> GenerateResponse:
        """Generate text with a normalized "stop" or "length" finish reason.

        Wrap API/transport errors as BackendCommunicationError and timeouts
        as BackendTimeoutError; keep local setup and processing errors distinct.

        None omits the application token cap. If an allowance is required,
        derive it from a known model limit or request an explicit positive one.
        Provider and model limits still apply.

        Native reasoning is backend configuration. Reject unsupported keywords,
        including per-call think.
        """

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> GenerateResponse:
        """Forward one user message to generate() without changing its content.

        Override this method to support conversations, other roles, or additional
        message fields; the plain-prompt adapter cannot preserve those semantics.
        """
        validate_nonblank_text(model, "model", optional=True)
        validate_generation_limits(max_new_tokens, 0, temperature=temperature, top_p=top_p)
        if "think" in kwargs:
            raise TypeError(
                "The per-call 'think' argument is no longer supported; "
                "configure native reasoning on the backend."
            )
        if not isinstance(messages, list) or any(not isinstance(msg, dict) for msg in messages):
            raise TypeError("messages must be a list of message dictionaries")
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError(
                "The default chat adapter accepts only one user message; "
                "implement generate_chat() to support other conversations."
            )
        message = messages[0]
        if set(message) - {"role", "content"}:
            raise ValueError(
                "The default chat adapter accepts only role and content fields; "
                "implement generate_chat() to support other message fields."
            )
        if not isinstance(message.get("content"), str):
            raise TypeError("User message content must be a string")
        return self.generate(
            message["content"],
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
