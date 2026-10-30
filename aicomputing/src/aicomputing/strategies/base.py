"""Strategy interface, shared prediction calls, and response accounting.

Strategies predict values without executing generated implementations. Native
reasoning follows backend configuration.

None leaves output uncapped by the application. Truncation retries double an
explicit allowance up to five times; provider limits still apply. Runtime
validation retries are separate.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from ..backends.base import Backend, GenerateResponse
from ..parsing import parse_result_detailed
from ..prompting import build_retry_prompt
from ..request import ExecuteRequest
from ..validation import ValidationMode

logger = logging.getLogger("aicomputing")

# Maximum additional calls after a response hits an explicit token budget.
# Each retry doubles that budget; uncapped requests cannot be expanded.
_MAX_RETRIES = 5


@dataclass
class CallStats:
    """Accumulate explicitly recorded responses, including all stages and retries.

    Token counts remain None until reported; zero means a reported zero.
    """

    model_calls: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    # Response-order snapshots: None = unrecorded, {} = omitted; not actual mode.
    reasoning_requests: List[Optional[dict[str, Any]]] = field(default_factory=list)

    # False once any recorded response omits the corresponding token count.
    # The counts above can therefore contain partial sums.
    input_complete: bool = True
    output_complete: bool = True

    # One snapshot per returned response; None means no observations recorded.
    reasoning_observations: List[Optional[dict[str, Any]]] = field(default_factory=list)
    transport_diagnostics: List[Optional[dict[str, Any]]] = field(default_factory=list)

    def record(self, response: "GenerateResponse") -> "CallStats":
        """Record usage, native requests, and response observations separately."""
        self.model_calls += 1

        self.reasoning_requests.append(
            deepcopy(getattr(response, "requested_reasoning", None))
        )
        self.reasoning_observations.append(
            deepcopy(getattr(response, "observed_reasoning", None))
        )
        self.transport_diagnostics.append(deepcopy(getattr(response, "transport_diagnostics", None)))
        raw_in = getattr(response, "input_tokens", None)
        raw_out = getattr(response, "output_tokens", None)
        if raw_out is None:
            raw_out = getattr(response, "tokens_used", None)
        if raw_in is None:
            self.input_complete = False
        if raw_out is None:
            self.output_complete = False

        self.input_tokens = _add_optional(self.input_tokens, raw_in)
        self.output_tokens = _add_optional(self.output_tokens, raw_out)
        return self

    @property
    def total_tokens(self) -> Optional[int]:
        """Return total usage, or None if either component is missing/incomplete."""
        if not (self.input_complete and self.output_complete):
            return None
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


def _add_optional(a: Optional[int], b: Optional[int]) -> Optional[int]:
    """Add reported counts, retaining a partial sum when one is unknown."""
    if a is None:
        return b
    if b is None:
        return a
    return a + b


@dataclass
class StrategyResult:
    """Result from a strategy execution (before runtime validation)."""
    value: Any
    raw_text: str

    # Stage-order prompts; repeated samples/token retries are not separate entries.
    prompts: List[str] = field(default_factory=list)

    tokens_used: Optional[int] = None

    # Final allowance, or None if uncapped/unrecorded. Multiple stages/samples
    # report their maximum unless any was uncapped.
    token_budget_used: Optional[int] = None
    retries: int = 0

    # Strategy selected by hybrid, when applicable.
    substrategy: Optional[str] = None

    # Original reason for the selected response, not a complete call history.
    provider_finish_reason: Optional[str] = None

    # Accumulated model-call accounting, when supplied by the strategy.
    call_stats: Optional[CallStats] = None

    # Parsing status and diagnostic for the selected answer.
    parse_status: Optional[str] = None
    parse_error: Optional[str] = None

    # Contract checker verdict parsing, separate from answer parsing.
    verdict_parse_status: Optional[str] = None

    # Whether revision changed the candidate; None if no check was reported.
    revised: Optional[bool] = None

    # Original candidate before revision, when applicable.
    pre_revision_value: Any = None


class Strategy(ABC):
    """Abstract base for execution strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'direct', 'cot', 'code_gen'."""

    def cache_state(self) -> dict[str, Any]:
        """Describe cache configuration; unsupported values bypass caching.

        Represent delegates by their configuration rather than strategy objects.
        """
        return vars(self).copy()

    @abstractmethod
    def execute(
        self,
        request: ExecuteRequest,
        backend: Backend,
        feedback: Optional[dict] = None,
    ) -> StrategyResult:
        """Predict with the supplied backend.

        Retry feedback carries the previous response and validation message
        under "raw_output" and "error".
        """


def _execute_prompt(
    request: ExecuteRequest,
    backend: Backend,
    prompt: str,
    feedback: Optional[dict] = None,
) -> StrategyResult:
    """Generate and parse one prediction prompt, preserving retries and accounting."""
    if feedback:
        prompt = build_retry_prompt(
            original_prompt=prompt,
            raw_output=feedback.get("raw_output", ""),
            error_message=feedback.get("error", ""),
            answer_format=request.answer_format,
        )

    initial_budget = request.max_new_tokens

    def call(max_new_tokens: Optional[int]):
        return backend.generate(
            prompt,
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=max_new_tokens,
        )

    psink: dict = {}
    parse = make_parser(request.return_type, _is_strict(request), psink, answer_format=request.answer_format)

    stats = CallStats()
    value, response, retries, final_budget = generate_with_retry(
        call=call,
        parse=parse,
        initial_budget=initial_budget,
        model_name=request.model or backend.default_model,
        stats=stats,
    )

    return StrategyResult(
        value=value,
        raw_text=response.text,
        prompts=[prompt],
        tokens_used=response.tokens_used,
        provider_finish_reason=response.provider_finish_reason,
        token_budget_used=final_budget,
        retries=retries,
        call_stats=stats,
        parse_status=psink.get("parse_status"),
        parse_error=psink.get("parse_error"),
    )


def generate_with_retry(
    call: Callable[[Optional[int]], GenerateResponse],
    parse: Callable[[str], Any],
    initial_budget: Optional[int],
    stats: Optional["CallStats"] = None,
    model_name: str = "",
) -> tuple[Any, GenerateResponse, int, Optional[int]]:
    """Double explicit token allowances after finish_reason="length".

    Uncapped calls cannot expand provider limits and are not retried here.
    Record every returned response in stats. Never parse truncated text;
    parse other responses once and propagate parser exceptions.

    Return (value, final_response, additional_calls, final_requested_budget).
    A parsed None may be valid; callers track recovery status separately.
    """
    if initial_budget is not None:
        if type(initial_budget) is not int:
            raise TypeError("initial_budget must be an integer or None")
        if initial_budget <= 0:
            raise ValueError("initial_budget must be positive")

    budget = initial_budget
    last_response = None
    # Preserve the last requested limit separately from the next budget.
    last_requested = budget

    for attempt in range(_MAX_RETRIES + 1):
        last_requested = budget
        response = call(budget)
        last_response = response
        # Include usage from responses discarded because of truncation.
        if stats is not None:
            stats.record(response)

        cutoff = response.finish_reason == "length"

        if not cutoff:
            value = parse(response.text)
            if attempt > 0:
                logger.info(
                    "[retry] %s: response not marked as truncated "
                    "at budget=%d (attempt %d)",
                    model_name, budget, attempt + 1,
                )
            return value, response, attempt, budget

        if budget is None:
            logger.warning(
                "[retry] %s: response reached a provider output or context "
                "limit without an application token cap; no larger budget "
                "can be requested",
                model_name,
            )
            return None, response, 0, None

        if attempt == _MAX_RETRIES:
            break

        new_budget = budget * 2
        logger.warning(
            "[retry] %s: hit token limit at budget=%d, "
            "retrying with budget=%d (retry %d/%d)",
            model_name, budget, new_budget, attempt + 1, _MAX_RETRIES,
        )
        budget = new_budget

    logger.error(
        "[retry] %s: all %d retries exhausted. "
        "Last budget=%d, finish_reason=%s",
        model_name, _MAX_RETRIES, last_requested,
        getattr(last_response, "finish_reason", "?"),
    )
    return None, last_response, _MAX_RETRIES, last_requested


def _same_value(a: Any, b: Any) -> bool:
    """Compare exact types and contents, ignoring dictionary insertion order.

    Sequence order matters; set order does not. Scalar values use their
    normal equality operation, without string or repr fallbacks.
    """
    if type(a) is not type(b):
        return False

    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(
            _same_value(left, right) for left, right in zip(a, b)
        )

    if isinstance(a, dict):
        return len(a) == len(b) and all(
            any(
                _same_value(left_key, right_key)
                and _same_value(left_value, right_value)
                for right_key, right_value in b.items()
            )
            for left_key, left_value in a.items()
        )

    if isinstance(a, (set, frozenset)):
        return len(a) == len(b) and all(
            any(_same_value(left, right) for right in b)
            for left in a
        )

    return bool(a == b)


def _accumulate(total: "CallStats", part: "Optional[CallStats]") -> "CallStats":
    """Merge an attempt's reported accounting into total, in place."""
    if part is None:
        # No accounting is available; leave the aggregate unchanged.
        return total
    total.model_calls += part.model_calls
    total.input_tokens = _add_optional(total.input_tokens, part.input_tokens)
    total.output_tokens = _add_optional(total.output_tokens, part.output_tokens)
    total.reasoning_requests.extend(deepcopy(part.reasoning_requests))
    total.reasoning_observations.extend(deepcopy(part.reasoning_observations))
    total.transport_diagnostics.extend(deepcopy(part.transport_diagnostics))
    # Any incomplete component keeps the aggregate component incomplete.
    total.input_complete = total.input_complete and part.input_complete
    total.output_complete = total.output_complete and part.output_complete
    return total


def make_parser(return_type, strict: bool, sink: dict, *, answer_format="legacy"):
    """Parse values and replace sink's parse_status/parse_error on each call."""
    def parse(text: str):
        outcome = parse_result_detailed(text, return_type, strict=strict, answer_format=answer_format)
        sink["parse_status"] = outcome.status
        sink["parse_error"] = outcome.error
        return outcome.value

    return parse


def _is_strict(request) -> bool:
    """Resolve validation_mode, then strict_types, otherwise WARN.

    Retained for parser compatibility; conversion rules are policy-independent.
    """
    mode = getattr(request, "validation_mode", None)
    if mode is None:
        mode = getattr(request, "strict_types", None)

    return ValidationMode.coerce(mode, ValidationMode.WARN) is ValidationMode.STRICT
