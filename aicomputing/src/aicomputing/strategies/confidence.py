"""
Confidence-gated strategy (self-assessed reasoning).

A single prompt, a single call: the model decides for itself, per task,
whether to answer immediately or trace its reasoning first. See the
module docstring in prompting.py (CONFIDENCE_SYSTEM) for the rationale —
in short, this replaces `hybrid`'s hand-written keyword classifier with a
decision the model makes about its own confidence, so the routing logic
has nothing to maintain and nothing to go stale as models improve.

PROVISIONAL: like `hybrid`, this is an unvalidated candidate strategy, not
a benchmarked-and-chosen default. It trades hybrid's cheap determinism for
reliance on the model's self-calibration, which may be poorly calibrated
on small/local models (e.g. always claiming confidence). Worth measuring
directly — e.g. does the model actually skip the trace on easy tasks and
use it on hard ones, and does that ratio shift with model capability.

Token budget: always sized for the "thinking" case (8192, doubling on
truncation) since the model may choose to trace at any point — we can't
know in advance which path it takes, unlike `direct` (never traces) or
`cot` (always traces).
"""

from __future__ import annotations

from typing import Optional

from ..backends.base import Backend
from ..model_config import get_token_budget
from ..parsing import parse_result
from ..prompting import build_confidence_prompt, build_retry_prompt
from ..request import ExecuteRequest
from .base import (CallStats, Strategy, StrategyResult,
                   generate_with_retry, make_parser, _is_strict)


class ConfidenceStrategy(Strategy):
    """Let the model decide, per call, whether it needs to reason first."""

    def __init__(self, include_fewshot: bool = True) -> None:
        self._include_fewshot = include_fewshot

    @property
    def name(self) -> str:
        return "confidence"

    def execute(
        self,
        request: ExecuteRequest,
        backend: Backend,
        feedback: Optional[dict] = None,
    ) -> StrategyResult:
        prompt = build_confidence_prompt(
            signature=request.signature,
            docstring=request.docstring,
            inputs=request.inputs,
            include_fewshot=self._include_fewshot,
        )
        if feedback:
            prompt = build_retry_prompt(
                original_prompt=prompt,
                raw_output=feedback.get("raw_output", ""),
                error_message=feedback.get("error", ""),
            )

        think = request.think if request.think is not None else True
        # Budget sizing always assumes the model might trace, regardless of
        # the native `think` toggle above — even non-thinking models can
        # still write a visible trace in the completion text (as `cot`
        # already does for them).
        initial_budget = request.max_new_tokens or get_token_budget(
            request.model or backend.default_model, True,
        )

        def call(max_new_tokens: int):
            return backend.generate(
                prompt,
                model=request.model,
                temperature=request.temperature,
                top_p=request.top_p,
                max_new_tokens=max_new_tokens,
                think=think,
            )

        psink: dict = {}
        parse = make_parser(request.return_type, _is_strict(request), psink)

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
