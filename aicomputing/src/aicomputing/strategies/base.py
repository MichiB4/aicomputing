"""
Abstract strategy interface for AI Computing.

A strategy defines *how* to turn a specification (signature + docstring + inputs)
into a predicted output value. Different strategies may use different prompts,
reasoning approaches, or even generate and execute code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from ..backends.base import Backend, GenerateResponse
from ..request import ExecuteRequest

logger = logging.getLogger("aicomputing")

# Doubling retry config. The sequence depends on the starting budget and is
# clamped by the cap, so it is not simply initial * 2**_MAX_RETRIES:
#   from   512: 512→1024→2048→4096→8192→16384  (5 doublings, cap not reached)
#   from  8192: 8192→16384→32768               (capped after 2 doublings)
_MAX_RETRIES = 5          # max doublings attempted
_MAX_TOKEN_CAP = 32768    # never exceed this regardless of doublings


@dataclass
class CallStats:
    """
    Running total of what an invocation actually cost the provider.

    Shared by every model call a strategy makes, including ones the caller
    never sees: budget-doubling retries inside generate_with_retry, contract's
    check stage, and each of consistency's samples. Passing one instance
    through all of them makes the sum automatic, so a multi-stage strategy
    cannot under-report by accounting for only its final response -- which is
    precisely the call that makes the cheap arms look comparable to the
    expensive ones.

    Token fields stay None until a provider actually reports them. A provider
    that reports nothing yields None rather than 0, because 0 would read as
    "this was free" in a cost table.
    """

    model_calls: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    # Thinking actually used, folded across every call of the invocation.
    # effective_think is True if ANY call reasoned, and thinking_honoured is
    # False if ANY call could not honour what the strategy requested -- both
    # in the conservative direction, so a Direct cell that reasoned even once
    # is visible rather than averaged away.
    requested_think: Optional[bool] = None
    effective_think: Optional[bool] = None
    thinking_honoured: Optional[bool] = None

    # Whether every call so far reported the component. A single call that did
    # not report its prompt tokens makes the INPUT side of the running sum
    # incomplete, and therefore makes the total unknowable -- summing the
    # calls that did report would silently treat the unreported ones as free.
    input_complete: bool = True
    output_complete: bool = True

    def record(self, response: "GenerateResponse") -> "CallStats":
        """Count one model call and add whatever usage it reported."""
        self.model_calls += 1
        req = getattr(response, "requested_think", None)
        eff = getattr(response, "effective_think", None)
        hon = getattr(response, "thinking_honoured", None)
        raw_in = getattr(response, "input_tokens", None)
        raw_out = getattr(response, "output_tokens", None)
        if raw_out is None:
            raw_out = getattr(response, "tokens_used", None)
        if raw_in is None:
            self.input_complete = False
        if raw_out is None:
            self.output_complete = False

        if req is not None:
            self.requested_think = req if self.requested_think is None else (
                self.requested_think or req)
        if eff is not None:
            self.effective_think = eff if self.effective_think is None else (
                self.effective_think or eff)
        if hon is not None:
            self.thinking_honoured = hon if self.thinking_honoured is None else (
                self.thinking_honoured and hon)
        self.input_tokens = _add_optional(self.input_tokens,
                                          getattr(response, "input_tokens", None))
        out = getattr(response, "output_tokens", None)
        if out is None:
            out = getattr(response, "tokens_used", None)
        self.output_tokens = _add_optional(self.output_tokens, out)
        return self

    @property
    def total_tokens(self) -> Optional[int]:
        """
        The total, or None when it is not genuinely known.

        This used to return (input or 0) + (output or 0) whenever EITHER side
        was reported, which silently treated an unknown input side as zero. A
        provider that reports only completion tokens then produced a total
        equal to its output count -- a plausible number that understated real
        cost by the whole prompt, on precisely the arms with the longest
        prompts.

        Both components must be known, for every call that contributed.
        """
        if not (self.input_complete and self.output_complete):
            return None
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


def _add_optional(a: Optional[int], b: Optional[int]) -> Optional[int]:
    """Sum two counts where None means 'not reported' rather than zero."""
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

    # The prompts actually sent, in order, for this execution.
    #
    # Captured rather than reconstructed. The benchmark scripts previously
    # rebuilt the logged prompt from the strategy name, which could not be
    # faithful: it fell through to the direct prompt for any strategy it did
    # not know (so `pseudocode` logged a prompt it never sent), it could not
    # represent a strategy that sends more than one (`contract` logged only
    # its first stage), and it had no way to know a validation retry had
    # wrapped the prompt in build_retry_prompt. Recording what was sent
    # removes all three, and the next strategy added gets correct logging for
    # free instead of silently inheriting the direct prompt.
    #
    # A list because the count is a property of the strategy: one for the
    # single-call arms, two for `contract`, and for `consistency` the one
    # prompt it samples n_samples times.
    prompts: List[str] = field(default_factory=list)

    tokens_used: Optional[int] = None
    # Final max_new_tokens ceiling requested, after any budget doubling.
    token_budget_used: Optional[int] = None
    retries: int = 0
    # Set only by strategies that delegate to another strategy (hybrid).
    substrategy: Optional[str] = None

    # The provider's own finish_reason, before it was normalised onto
    # {"stop", "length"}. Carried for audit: an unrecognised value — including
    # a tool-use one — is otherwise logged once and collapsed to "stop", which
    # nobody sees in a run of thousands. Strategies making several calls report
    # the response they otherwise surface, so this is a sample, not a census.
    provider_finish_reason: Optional[str] = None

    # Set only by the code-generation baseline, which is the one component
    # that executes what the model produced instead of predicting from the
    # spec. "ok" | "error" | "timeout"; None for every spec-only strategy,
    # which has no execution step to fail. Carried so that the baseline's
    # execution-failure rate is reportable next to its accuracy rather than
    # folded into it — a generation that does not run scores as wrong.
    exec_status: Optional[str] = None

    # Cost accounting across every model call this strategy made. None only
    # for results built without going through generate_with_retry.
    call_stats: Optional[CallStats] = None

    # How the final value was recovered from the model's text.
    parse_status: Optional[str] = None
    parse_error: Optional[str] = None

    # Contract only: how the VERIFIER's verdict object parsed. Kept apart
    # from parse_status, which always describes the returned answer.
    verdict_parse_status: Optional[str] = None

    # Set only by strategies that re-examine their own answer (contract).
    # None means the strategy has no revision step, which is different from
    # False, meaning it had one and kept the original value.
    revised: Optional[bool] = None
    # The value before the revision step. Kept so that flip rate — whether a
    # revision corrected a wrong answer or corrupted a right one — is
    # recoverable per example; the final value alone cannot distinguish them.
    pre_revision_value: Any = None


class Strategy(ABC):
    """Abstract base for execution strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'direct', 'cot', 'code_gen'."""

    @abstractmethod
    def execute(
        self,
        request: ExecuteRequest,
        backend: Backend,
        feedback: Optional[dict] = None,
    ) -> StrategyResult:
        """
        Execute the AI function call using this strategy.

        Receives the full request and a backend to generate text.
        `feedback` is set on validation retries and contains the previous
        raw output and the validation error message:
        {"raw_output": str, "error": str}. Strategies append it to the
        prompt so the model can correct its answer.
        Returns a StrategyResult with the parsed value.
        """


def generate_with_retry(
    call: Callable[[int], GenerateResponse],
    parse: Callable[[str], Any],
    initial_budget: int,
    stats: Optional["CallStats"] = None,
    model_name: str = "",
) -> tuple[Any, GenerateResponse, int, int]:
    """
    Call the backend with doubling retries when the model hits the token limit.

    Args:
        call:           Function that takes max_new_tokens and returns GenerateResponse.
        parse:          Function that takes the response text and returns the parsed value.
                        Should raise ValueError or return None on failure.
        initial_budget: Starting token limit (512 for no-think, 8192 for think).
        model_name:     Used only for log messages.

    Returns:
        (parsed_value, final_response, retries, final_budget) — retries is the
        number of times the budget was doubled (0 means it worked on the first
        attempt); final_budget is the max_new_tokens ceiling actually requested
        on the last attempt. final_budget is returned rather than re-derived by
        callers because the doubling is capped, so it is not simply
        initial_budget * 2**retries.
    """
    budget = initial_budget
    last_response = None
    # The ceiling actually handed to the backend on the most recent attempt.
    # Tracked separately from `budget`, which is doubled at the END of a
    # truncated iteration and therefore runs one step ahead of what was really
    # requested. Returning `budget` on the loop-exhaustion path reported a
    # ceiling that was never used — from a 512 start the requests end at 16384
    # while `budget` has already become 32768 — and that value reaches
    # ExecuteResult.token_budget_used and the trace, i.e. the one field whose
    # whole purpose is to say what ceiling produced this answer.
    last_requested = initial_budget

    for attempt in range(_MAX_RETRIES + 1):
        last_requested = budget
        response = call(budget)
        last_response = response
        # Counted here, inside the loop, so a truncation retry is billed. The
        # earlier attempts really were sent and really were charged; counting
        # only the survivor would make a model that needs three attempts look
        # as cheap as one that answers first time.
        if stats is not None:
            stats.record(response)

        cutoff = response.finish_reason == "length"

        if not cutoff:
            try:
                value = parse(response.text)
                if attempt > 0:
                    logger.info(
                        "[retry] %s: succeeded at budget=%d (attempt %d)",
                        model_name, budget, attempt + 1,
                    )
                return value, response, attempt, budget
            except (ValueError, TypeError):
                return None, response, attempt, budget

        if budget >= _MAX_TOKEN_CAP:
            # Already tried at the cap — retrying at the same budget
            # would give the same truncated output.
            logger.error(
                "[retry] %s: still truncated at token cap %d, giving up.",
                model_name, _MAX_TOKEN_CAP,
            )
            return None, last_response, attempt, budget

        new_budget = min(budget * 2, _MAX_TOKEN_CAP)
        logger.warning(
            "[retry] %s: hit token limit at budget=%d, retrying with budget=%d (attempt %d/%d)",
            model_name, budget, new_budget, attempt + 1, _MAX_RETRIES,
        )
        budget = new_budget

    logger.error(
        "[retry] %s: all %d retries exhausted. Last budget=%d, finish_reason=%s",
        model_name, _MAX_RETRIES, last_requested,
        getattr(last_response, "finish_reason", "?"),
    )
    return None, last_response, _MAX_RETRIES, last_requested


def _accumulate(total: "CallStats", part: "Optional[CallStats]") -> "CallStats":
    """
    Fold one attempt's usage into a running total for the whole invocation.

    Kept next to CallStats so the None-means-unreported rule is applied in one
    place: a provider that reports nothing must leave the total None rather
    than contributing a 0 that would read as "this attempt was free".
    """
    if part is None:
        return total
    total.model_calls += part.model_calls
    total.input_tokens = _add_optional(total.input_tokens, part.input_tokens)
    total.output_tokens = _add_optional(total.output_tokens, part.output_tokens)
    for field in ("requested_think", "effective_think"):
        other = getattr(part, field)
        if other is not None:
            cur = getattr(total, field)
            setattr(total, field, other if cur is None else (cur or other))
    # Incompleteness is contagious: one call with an unknown component makes
    # the aggregate total unknown, however many other calls reported fully.
    total.input_complete = total.input_complete and part.input_complete
    total.output_complete = total.output_complete and part.output_complete
    if part.thinking_honoured is not None:
        total.thinking_honoured = (part.thinking_honoured
                                   if total.thinking_honoured is None
                                   else (total.thinking_honoured and part.thinking_honoured))
    return total


def make_parser(return_type, strict: bool, sink: dict):
    """
    Build the `parse` callable for generate_with_retry, recording HOW it parsed.

    generate_with_retry's parse hook returns a value, so the status has
    nowhere to go in its signature. Rather than change that signature at six
    call sites, the outcome is written into `sink`, which the strategy then
    attaches to its StrategyResult. The last call wins, which is the right
    one: on a retried invocation the status describes the answer actually
    returned.
    """
    from ..parsing import parse_result_detailed

    def parse(text: str):
        outcome = parse_result_detailed(text, return_type, strict=strict)
        sink["parse_status"] = outcome.status
        sink["parse_error"] = outcome.error
        return outcome.value

    return parse


def _is_strict(request) -> bool:
    """
    Whether this request runs under STRICT structural validation.

    Read here so coercion inside the parser matches the policy the runtime
    will apply afterwards. If the parser repaired a value that STRICT would
    then reject, the two halves of the policy would disagree and the model
    would be credited with output it did not produce.
    """
    from ..validation import ValidationMode
    mode = getattr(request, "validation_mode", None)
    if mode is None:
        mode = getattr(request, "strict_types", None)
    return ValidationMode.coerce(mode, ValidationMode.WARN) is ValidationMode.STRICT
