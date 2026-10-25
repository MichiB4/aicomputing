"""
Abstract strategy interface for AI Computing.

Registered strategies predict outputs from specifications and inputs without
executing generated implementations. The code-generation comparison baseline
also implements this interface but is supplied explicitly outside the registry.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from ..backends.base import Backend, GenerateResponse
from ..parsing import parse_result_detailed
from ..request import ExecuteRequest
from ..validation import ValidationMode

logger = logging.getLogger("aicomputing")

# Maximum additional calls after a response hits an explicit token budget.
# Each retry doubles that budget; uncapped requests cannot be expanded.
_MAX_RETRIES = 5


@dataclass
class CallStats:
    """
    Accumulate model-call counts, reported usage, and requested settings.

    Strategies share one instance across their calls, including token-budget
    retries, contract checks, and consistency samples. Each response must be
    recorded explicitly.

    Token counts remain None until reported. Zero represents a reported
    count of zero.
    """

    model_calls: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    # One snapshot per recorded response, in call order, including retries.
    # None means unrecorded settings; {} means no native override was sent.
    # These records do not establish the provider's actual reasoning mode.
    reasoning_requests: List[Optional[dict[str, Any]]] = field(default_factory=list)

    # False once any recorded response omits the corresponding token count.
    # The counts above can therefore contain partial sums.
    input_complete: bool = True
    output_complete: bool = True

    # One snapshot per returned response; None means no observations recorded.
    reasoning_observations: List[Optional[dict[str, Any]]] = field(default_factory=list)

    def record(self, response: "GenerateResponse") -> "CallStats":
        """Record usage, native requests, and response observations separately."""
        self.model_calls += 1

        self.reasoning_requests.append(
            deepcopy(getattr(response, "requested_reasoning", None))
        )
        self.reasoning_observations.append(
            deepcopy(getattr(response, "observed_reasoning", None))
        )
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
        """
        Return total token usage when both components are complete.

        Return None if any recorded response omitted input or output usage,
        or if no counts have been reported.
        """
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

    # Prompt texts recorded by the strategy, in stage order.
    # Repeated sampling and token-budget retries are not separate entries.
    prompts: List[str] = field(default_factory=list)

    tokens_used: Optional[int] = None

    # Final requested output-token limit; None means no application cap.
    # Contract and consistency report None if any stage/sample was uncapped,
    # otherwise the maximum final limit. Custom strategies may leave it unset.
    token_budget_used: Optional[int] = None
    retries: int = 0

    # Strategy selected by hybrid, when applicable.
    substrategy: Optional[str] = None

    # Original provider finish reason for the response surfaced by the strategy.
    # This is not a complete history for strategies making multiple calls.
    provider_finish_reason: Optional[str] = None

    # Code-generation/execution status, including "truncated" before execution.
    # None when no execution status is supplied.
    exec_status: Optional[str] = None

    # Accumulated model-call accounting, when supplied by the strategy.
    call_stats: Optional[CallStats] = None

    # Parsing status and diagnostic for the selected answer.
    parse_status: Optional[str] = None
    parse_error: Optional[str] = None

    # Contract checker verdict parsing, separate from answer parsing.
    verdict_parse_status: Optional[str] = None

    # Whether a revision step changed the candidate value.
    # None means no revision step is reported.
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
        """Describe configuration for the runtime's cache key.

        Unsupported values cause the cache to bypass the request.
        Strategies owning other strategies can describe their delegates
        explicitly instead of including the delegate objects themselves.
        """
        return vars(self).copy()

    @abstractmethod
    def execute(
        self,
        request: ExecuteRequest,
        backend: Backend,
        feedback: Optional[dict] = None,
    ) -> StrategyResult:
        """
        Produce a prediction and diagnostics using the supplied backend.

        On validation retries, feedback contains the previous raw response
        and validation message under "raw_output" and "error". Strategies
        use this information when constructing their retry prompts.
        """


def generate_with_retry(
    call: Callable[[Optional[int]], GenerateResponse],
    parse: Callable[[str], Any],
    initial_budget: Optional[int],
    stats: Optional["CallStats"] = None,
    model_name: str = "",
) -> tuple[Any, GenerateResponse, int, Optional[int]]:
    """
    Retry token-limited responses with increasing output-token budgets.

    Only responses with finish_reason="length" and an explicit integer
    budget trigger a retry. None requests no application cap, so a provider
    output or context limit cannot be expanded here and is not retried.
    Truncated prefixes are never parsed. Other responses are parsed once.
    Parser exceptions propagate unchanged.
    A parser may return None as a valid value; this function does not
    distinguish it from a returned None representing failed recovery.

    Args:
        call: Generate a response using the supplied output-token budget.
        parse: Recover a Python value from response text.
        initial_budget: Positive starting budget, or None for no application cap.
            Retries may exceed an explicit starting budget.
        stats: Optional accumulator for each returned backend response.
        model_name: Model identifier used in log messages.

    Returns:
        A tuple of (value, final_response, retries, final_budget).
        retries counts additional calls after the first, regardless of
        success. final_budget is the limit requested for the final call,
        or None when no application cap was requested.
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
    # Any incomplete component keeps the aggregate component incomplete.
    total.input_complete = total.input_complete and part.input_complete
    total.output_complete = total.output_complete and part.output_complete
    return total


def make_parser(return_type, strict: bool, sink: dict):
    """
    Build a parser callback that stores recovery diagnostics in sink.

    Each completed parse replaces sink's parse_status and parse_error.
    The caller associates those diagnostics with the corresponding value.
    """
    def parse(text: str):
        outcome = parse_result_detailed(text, return_type, strict=strict)
        sink["parse_status"] = outcome.status
        sink["parse_error"] = outcome.error
        return outcome.value

    return parse


def _is_strict(request) -> bool:
    """
    Resolve the request's STRICT flag for existing parser call sites.

    Prefer validation_mode, then strict_types, and otherwise use WARN.
    Runtime supplies its resolved output policy in validation_mode.
    Limited conversion rules no longer depend on this flag.
    """
    mode = getattr(request, "validation_mode", None)
    if mode is None:
        mode = getattr(request, "strict_types", None)

    return ValidationMode.coerce(mode, ValidationMode.WARN) is ValidationMode.STRICT
