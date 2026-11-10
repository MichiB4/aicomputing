"""Predict an answer, then ask the model to accept or revise it.

The initial stage uses the direct prompt. The check stage returns a complete
verdict object; malformed checks retain the candidate. Explicit replacements,
including None, are validated by the runtime under the normal output policy.
Parsing metadata describes the selected answer. Usage includes both stages;
model self-checking does not prove correctness.
Both stages use the same backend configuration for native reasoning.
Neither stage imposes an application output-token cap unless the request
supplies one. Provider output and context limits still apply.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ..backends.base import Backend
from ..parsing import ParseOutcome, coerce_type, parse_result
from ..prompting import build_contract_check_prompt, build_direct_prompt
from ..request import ExecuteRequest
from ..validation import validate_bool
from .base import (
    CallStats,
    Strategy,
    StrategyResult,
    _backend_call,
    _output_type,
    _same_value,
    _with_feedback,
    _generate_once,
    make_parser,
)


class ContractStrategy(Strategy):
    """Predict with `direct`, then test the candidate against the docstring."""

    def __init__(self, include_fewshot: bool = True) -> None:
        validate_bool(include_fewshot, "include_fewshot")
        self._include_fewshot = include_fewshot

    @property
    def name(self) -> str:
        return "contract"

    def execute(
        self,
        request: ExecuteRequest,
        backend: Backend,
        feedback: Optional[dict] = None,
    ) -> StrategyResult:
        return_type = _output_type(request)

        # Keep prediction and verdict parsing diagnostics separate.
        # Final answer metadata is selected after applying the verdict.
        parse = make_parser(return_type)

        def parse_verdict(text: str) -> ParseOutcome:
            return parse_result(text, unwrap_result=False)

        # Stage 1: predict a candidate using the direct prompt.
        predict_prompt = _with_feedback(build_direct_prompt(
            signature=request.signature,
            docstring=request.docstring,
            inputs=request.inputs,
            include_fewshot=self._include_fewshot,
        ), feedback)

        # Both stages use the same requested limit, including None.
        stats = CallStats()
        candidate_outcome, predict_response = _generate_once(
            call=_backend_call(request, backend, predict_prompt),
            parse=parse,
            stats=stats,
        )
        candidate = candidate_outcome.value

        # Stage 2: ask the model to accept or revise the candidate.
        check_prompt = build_contract_check_prompt(
            signature=request.signature,
            docstring=request.docstring,
            inputs=request.inputs,
            candidate=candidate,
            include_fewshot=self._include_fewshot,
        )

        verdict_outcome, check_response = _generate_once(
            call=_backend_call(request, backend, check_prompt),
            parse=parse_verdict,
            stats=stats,
        )
        verdict_object = verdict_outcome.value

        final_value, revised = _apply_verdict(
            candidate, verdict_object, return_type=return_type,
        )

        # An explicit replacement supplies a recovered answer even when it
        # equals the candidate, including None after failed recovery.
        # Runtime type and constraint validation still apply.
        replacement_supplied = (
            isinstance(verdict_object, dict)
            and verdict_object.get("verdict") == "revise"
            and "result" in verdict_object
        )
        selected_outcome = verdict_outcome if replacement_supplied else candidate_outcome
        selected_response = check_response if replacement_supplied else predict_response

        return StrategyResult(
            value=final_value,
            # Record prediction and check prompts in execution order.
            prompts=[predict_prompt, check_prompt],
            raw_text=(
                f"[contract {'changed' if revised else 'unchanged'}] "
                f"{predict_response.text}\n--- check ---\n{check_response.text}"
            ),
            feedback_text=(
                f"[contract {'changed' if revised else 'unchanged'}] "
                f"{predict_response.parsing_text}\n--- check ---\n{check_response.parsing_text}"
            ),
            provider_finish_reason=selected_response.provider_finish_reason,
            call_stats=stats,
            # Describe the selected answer's parsing outcome.
            parse_status=selected_outcome.status,
            parse_error=selected_outcome.error,
            verdict_parse_status=verdict_outcome.status,
            verdict_parse_error=verdict_outcome.error,
            verdict_finish_reason=check_response.provider_finish_reason,
            revised=revised,
            pre_revision_value=candidate,
        )




def _apply_verdict(
    candidate: Any,
    verdict_object: Any,
    *,
    return_type: Any = None,
) -> Tuple[Any, bool]:
    """Apply a replacement after the agreed output-type conversions.

    revised reports a change in type or contents after conversion;
    dictionary order is ignored. This does not establish that either
    value is correct.
    """
    if (
        not isinstance(verdict_object, dict)
        or verdict_object.get("verdict") != "revise"
        or "result" not in verdict_object
    ):
        return candidate, False

    converted = coerce_type(verdict_object["result"], return_type)
    if _same_value(converted, candidate):
        return candidate, False
    return converted, True
