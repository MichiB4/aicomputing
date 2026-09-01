"""
Abstract backend interface for AI Computing.

A backend handles the raw communication with an LLM: sending a prompt
and receiving generated text. Different backends talk to different
inference engines (Ollama, HuggingFace, OpenAI API, etc.) but expose
the same interface to the strategy layer.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..model_config import DEFAULT_MAX_NEW_TOKENS, DEFAULT_THINK

logger = logging.getLogger("aicomputing")


class ToolUseDetected(RuntimeError):
    """
    The provider ran a tool (typically a code interpreter) on our behalf.

    This invalidates the measurement rather than degrading it. The thesis's
    claim is that the model predicts a function's output by reasoning from the
    specification alone; if an interpreter produced the value, the number
    measures the interpreter. No strategy in this package ever declares tools,
    so this can only arise from a provider-side default — which is exactly the
    case that would otherwise pass unnoticed, because a tool call returns
    `content=None` and reads downstream as an ordinary parse failure.

    Raised rather than logged deliberately: a crashed run is recoverable, a
    silently contaminated results table defended at a viva is not.
    """


@dataclass
class GenerateResponse:
    """Raw response from a backend."""
    text: str
    model: str

    # Completion tokens. Historical name, kept because every backend and
    # strategy already reads it; output_tokens is the same number under the
    # name the result schema uses.
    tokens_used: Optional[int] = None

    # Prompt and completion tokens as reported by the provider. Both None when
    # the provider does not report usage -- stored as None rather than
    # estimated, so a cost table never contains an invented number.
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    # What the strategy asked for vs. what the model family actually does.
    # Populated from model_config.resolve_thinking(). None on backends that do
    # not model thinking. See ThinkingResolution for why the distinction is an
    # experimental confound and not a detail.
    requested_think: Optional[bool] = None
    effective_think: Optional[bool] = None
    thinking_honoured: Optional[bool] = None
    finish_reason: Optional[str] = None   # "stop" | "length" | None
    raw: Optional[Dict[str, Any]] = None  # backend-specific metadata

    # The provider's own finish_reason, before normalize_finish_reason() maps
    # it onto {"stop", "length"}. Kept because that mapping is lossy in the one
    # direction that matters for validity: an unrecognised value (including
    # "tool_calls") is logged once and collapsed to "stop", which is invisible
    # in a run of thousands. Recording it per call makes a post-hoc audit
    # possible from stored traces instead of only at call time.
    provider_finish_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# finish_reason normalisation
#
# The strategy layer decides whether to retry with a doubled token budget by
# testing `finish_reason == "length"` exactly (strategies/base.py). Providers
# do not agree on the spelling: OpenAI-style APIs say "length", Gemini says
# "MAX_TOKENS", Anthropic-style says "max_tokens". Passing a provider value
# through unmapped means truncation is never detected on those providers, and
# the truncated output is scored as a wrong answer instead of being retried.
#
# Every backend must therefore map its provider's value onto exactly
# {"stop", "length"} before constructing a GenerateResponse.
# ---------------------------------------------------------------------------

# Values meaning "the model was cut off by the token limit".
_LENGTH_ALIASES = {
    "length", "max_tokens", "maxtokens", "max_output_tokens",
    "model_length", "token_limit", "truncated",
}

# Values meaning "the model finished on its own terms". Includes reasons that
# are not truncation even though they are not a clean stop (a content filter
# or tool call must not trigger a budget retry — a larger budget won't help).
_STOP_ALIASES = {
    "stop", "end_turn", "eos", "eos_token", "stop_sequence",
    "complete", "completed", "finished", "content_filter",
    "tool_calls", "tool_use", "function_call",
}


# Response keys that indicate the provider executed something on our behalf.
# Matched against JSON *keys* only, never values — a model that merely writes
# the phrase "code execution" in its prose must not trip the guard.
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


def detect_tool_use(choice: Any, raw_dump: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Return a short description of provider-side tool use, or None if clean.

    Three independent signals, because providers disagree on how they report
    it and the set will keep growing:

    1. `message.tool_calls` / `message.function_call` — the OpenAI shape.
    2. A finish_reason meaning the model stopped to call something.
    3. A tool/execution key anywhere in the raw response body — the catch-all
       for provider-specific shapes not yet enumerated here.

    Only keys are inspected for (3). Scanning values would false-positive on
    any completion whose text happens to discuss code execution, which for
    this project is a large fraction of them.
    """
    message = getattr(choice, "message", None)
    if message is not None:
        if getattr(message, "tool_calls", None):
            return "message.tool_calls is populated"
        if getattr(message, "function_call", None):
            return "message.function_call is populated"

    finish = str(getattr(choice, "finish_reason", "") or "").strip().lower()
    if finish.replace("-", "_") in _TOOL_USE_FINISH_REASONS:
        return f"finish_reason={finish!r}"

    if raw_dump:
        for key in _iter_keys(raw_dump):
            normalised = key.strip().lower().replace("-", "_")
            if normalised in _TOOL_USE_KEYS:
                # A present-but-null key is the SDK serialising an unused
                # field; only a populated one is evidence.
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
    """
    Map a provider's finish_reason onto "stop" or "length".

    Comparison is case-insensitive and ignores separator style, so
    "MAX_TOKENS", "max-tokens" and "maxTokens" all resolve to "length".

    An unrecognised value is logged at WARNING and treated as "stop". "stop"
    is the conservative default: it costs a possible missed retry, whereas
    defaulting to "length" would burn a doubled-budget retry on every
    unknown-but-complete response.
    """
    if value is None:
        return "stop"

    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    # Collapse camelCase ("maxTokens" -> "maxtokens") for the alias lookup.
    flat = key.replace("_", "")

    if key in _LENGTH_ALIASES or flat in _LENGTH_ALIASES:
        return "length"
    if key in _STOP_ALIASES or flat in _STOP_ALIASES:
        return "stop"

    logger.warning(
        "Unrecognised finish_reason %r from model %r — treating as 'stop'. "
        "Truncation will not be retried for this response; add the value to "
        "_LENGTH_ALIASES or _STOP_ALIASES in backends/base.py if it recurs.",
        value, model or "<unknown>",
    )
    return "stop"


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
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        stop: Optional[List[str]] = None,
        think: bool = DEFAULT_THINK,
        **kwargs,
    ) -> GenerateResponse:
        """
        Send a prompt and return the generated text.

        Implementations must return a GenerateResponse whose finish_reason is
        already normalised to "stop" or "length" (see normalize_finish_reason);
        the strategy layer tests for "length" exactly.

        Defaults for max_new_tokens and think come from model_config so that
        every backend behaves identically when a caller omits them.
        """

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        think: bool = DEFAULT_THINK,
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
            think=think,
            **kwargs,
        )

    def is_available(self) -> bool:
        """Check if the backend is reachable / importable."""
        return True
