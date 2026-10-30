"""Predict an answer, then ask the model to accept or revise it.

The initial stage uses the direct prompt. The check stage returns a complete
verdict object; malformed checks retain the candidate. Explicit replacements,
including None, are validated by the runtime under the normal output policy.
Parsing metadata describes the selected answer. Usage includes both stages
and any token-budget retries; model self-checking does not prove correctness.
Both stages use the same backend configuration for native reasoning.
Neither stage imposes an application output-token cap unless the request
supplies one. Provider output and context limits still apply.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ..backends.base import Backend
from ..answer_protocol import FRAMED_TEXT_V1
from ..parsing import ParseOutcome, coerce_type, parse_result_detailed
from ..prompting import (
    build_contract_check_prompt,
    build_direct_prompt,
    build_retry_prompt,
)
from ..request import ExecuteRequest
from .base import (
    CallStats,
    Strategy,
    StrategyResult,
    _is_strict,
    _same_value,
    generate_with_retry,
    make_parser,
)


class ContractStrategy(Strategy):
    """Predict with `direct`, then test the candidate against the docstring."""

    def __init__(self, include_fewshot: bool = True) -> None:
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
        model_name = request.model or backend.default_model
        # Both stages inherit an explicit budget or leave generation uncapped.
        predict_budget_start = request.max_new_tokens
        check_budget_start = request.max_new_tokens

        def make_call(prompt: str):
            def call(max_new_tokens: Optional[int]):
                return backend.generate(
                    prompt,
                    model=request.model,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_new_tokens=max_new_tokens,
                )
            return call

        # Keep prediction and verdict parsing diagnostics separate.
        # Final answer metadata is selected after applying the verdict.
        psink: dict = {}
        vsink: dict = {}
        strict = _is_strict(request)
        parse = make_parser(request.return_type, strict, psink, answer_format=request.answer_format)

        def parse_verdict(text: str):
            outcome = _parse_verdict(text, answer_format=request.answer_format)
            vsink["parse_status"] = outcome.status
            vsink["parse_error"] = outcome.error
            return outcome.value

        # Stage 1: predict a candidate using the direct prompt.
        predict_prompt = build_direct_prompt(
            signature=request.signature,
            docstring=request.docstring,
            inputs=request.inputs,
            include_fewshot=self._include_fewshot,
            answer_format=request.answer_format,
        )
        if feedback:
            predict_prompt = build_retry_prompt(
                original_prompt=predict_prompt,
                raw_output=feedback.get("raw_output", ""),
                error_message=feedback.get("error", ""),
                answer_format=request.answer_format,
            )

        # Accumulate returned-response usage across both stages and retries.
        stats = CallStats()
        (
            candidate,
            predict_response,
            predict_retries,
            predict_budget,
        ) = generate_with_retry(
            call=make_call(predict_prompt),
            parse=parse,
            initial_budget=predict_budget_start,
            model_name=model_name,
            stats=stats,
        )

        # Stage 2: ask the model to accept or revise the candidate.
        check_prompt = build_contract_check_prompt(
            signature=request.signature,
            docstring=request.docstring,
            inputs=request.inputs,
            candidate=candidate,
            include_fewshot=self._include_fewshot,
            answer_format=request.answer_format,
        )

        (
            verdict_object,
            check_response,
            check_retries,
            check_budget,
        ) = generate_with_retry(
            call=make_call(check_prompt),
            parse=parse_verdict,
            initial_budget=check_budget_start,
            model_name=model_name,
            stats=stats,
        )

        final_value, revised = _apply_verdict(
            candidate, verdict_object, return_type=request.return_type,
        )

        # An explicit replacement supplies a recovered answer even when it
        # equals the candidate, including None after failed recovery.
        # Runtime type and constraint validation still apply.
        replacement_supplied = (
            isinstance(verdict_object, dict)
            and verdict_object.get("verdict") == "revise"
            and "result" in verdict_object
        )
        answer_parse_status = (
            vsink.get("parse_status", ParseOutcome.PARSED) if replacement_supplied
            else psink.get("parse_status")
        )
        answer_parse_error = (
            vsink.get("parse_error") if replacement_supplied else psink.get("parse_error")
        )

        tokens = _sum_optional(
            predict_response.tokens_used, check_response.tokens_used,
        )

        return StrategyResult(
            value=final_value,
            # Record prediction and check prompts in execution order.
            prompts=[predict_prompt, check_prompt],
            raw_text=(
                f"[contract {'changed' if revised else 'unchanged'}] "
                f"{predict_response.text}\n--- check ---\n{check_response.text}"
            ),
            tokens_used=tokens,
            provider_finish_reason=check_response.provider_finish_reason,
            # Any uncapped stage makes the combined allowance uncapped.
            token_budget_used=(
                None if predict_budget is None or check_budget is None
                else max(predict_budget, check_budget)
            ),
            retries=predict_retries + check_retries,
            call_stats=stats,
            # Describe the selected answer's parsing outcome.
            parse_status=answer_parse_status,
            parse_error=answer_parse_error,
            verdict_parse_status=vsink.get("parse_status"),
            revised=revised,
            pre_revision_value=candidate,
        )


def _parse_verdict(check_text: str, *, answer_format="legacy") -> ParseOutcome:
    """Recover and validate a complete contract verdict without unwrapping it."""
    outcome = parse_result_detailed(check_text, unwrap_result=False, answer_format=answer_format)
    recovered = outcome.status == ParseOutcome.PARSED or (
        answer_format == FRAMED_TEXT_V1 and outcome.status == ParseOutcome.FALLBACK
    )
    if not recovered:
        return ParseOutcome(
            None, ParseOutcome.FAILED,
            outcome.error or "no structured verdict found",
        )
    obj = outcome.value
    if not isinstance(obj, dict):
        return ParseOutcome(None, ParseOutcome.FAILED, "verdict must be an object")
    verdict = obj.get("verdict")
    if not isinstance(verdict, str) or verdict.lower() not in ("accept", "revise"):
        return ParseOutcome(None, ParseOutcome.FAILED, "missing or invalid verdict")
    verdict = verdict.lower()
    if verdict == "revise" and "result" not in obj:
        return ParseOutcome(None, ParseOutcome.FAILED, "revision has no result field")
    return ParseOutcome({**obj, "verdict": verdict}, outcome.status, outcome.error)


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

    checked = coerce_type(verdict_object["result"], return_type)
    if _same_value(checked, candidate):
        return candidate, False
    return checked, True


def _sum_optional(*values: Optional[int]) -> Optional[int]:
    """Sum token counts, returning None when no call reported one."""
    present = [v for v in values if v is not None]
    return sum(present) if present else None
