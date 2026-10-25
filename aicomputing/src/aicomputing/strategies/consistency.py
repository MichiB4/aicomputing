"""Self-consistency prediction using repeated structured-CoT samples.

Selects the most frequent recovered value matching the expected return type,
with ties resolved by first occurrence. Any leaves voting unrestricted.
None participates when its type is allowed; failed recovery does not vote.
The selected sample supplies answer text and parsing metadata, while
usage and token-budget retry counts are accumulated across samples.
Every sample requests temperature=0.0, regardless of the request setting.
Native reasoning is configured by the backend rather than selected by
this strategy.
Samples have no application output-token cap unless the request supplies one.
Provider output and context limits still apply.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from ..backends.base import Backend
from ..parsing import ParseOutcome
from ..prompting import build_cot_prompt, build_retry_prompt
from ..request import ExecuteRequest
from ..validation import validate_output
from .base import (
    CallStats,
    Strategy,
    StrategyResult,
    _is_strict,
    _same_value,
    generate_with_retry,
    make_parser,
)


class SelfConsistencyStrategy(Strategy):
    """Sample `n_samples` CoT traces and return the most frequent value."""

    def __init__(self, n_samples: int = 5, include_fewshot: bool = True) -> None:
        if type(n_samples) is not int:
            raise TypeError("n_samples must be an integer")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        self._n_samples = n_samples
        self._include_fewshot = include_fewshot

    @property
    def name(self) -> str:
        return "consistency"

    def execute(
        self,
        request: ExecuteRequest,
        backend: Backend,
        feedback: Optional[dict] = None,
    ) -> StrategyResult:
        prompt = build_cot_prompt(
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

        # None leaves output length to the provider rather than a strategy cap.
        initial_budget = request.max_new_tokens

        def call(max_new_tokens: Optional[int]):
            return backend.generate(
                prompt,
                model=request.model,
                temperature=0.0,
                top_p=request.top_p,
                max_new_tokens=max_new_tokens,
            )

        # Accumulate returned-response usage across samples and token retries.
        stats = CallStats()
        samples: List[StrategyResult] = []
        strict = _is_strict(request)
        for _ in range(self._n_samples):
            # A fresh sink prevents another sample from overwriting the
            # parsing status of the answer eventually selected by voting.
            psink: dict = {}
            parse = make_parser(request.return_type, strict, psink)
            value, response, retries, final_budget = generate_with_retry(
                call=call,
                parse=parse,
                initial_budget=initial_budget,
                model_name=request.model or backend.default_model,
                stats=stats,
            )
            samples.append(StrategyResult(
                value=value,
                raw_text=response.text,
                tokens_used=response.tokens_used,
                provider_finish_reason=response.provider_finish_reason,
                token_budget_used=final_budget,
                retries=retries,
                parse_status=psink.get("parse_status"),
                parse_error=psink.get("parse_error"),
            ))

        winner, vote_count = _majority_vote(
            samples, return_type=request.return_type,
        )
        reported_tokens = [
            sample.tokens_used
            for sample in samples
            if sample.tokens_used is not None
        ]
        total_tokens = sum(reported_tokens) if reported_tokens else None
        total_retries = sum(sample.retries for sample in samples)
        # Any uncapped sample makes the combined allowance uncapped.
        budgets = [sample.token_budget_used for sample in samples]
        max_budget = (
            None if any(budget is None for budget in budgets) else max(budgets)
        )

        return StrategyResult(
            value=winner.value,
            # All samples use this prompt; record it once.
            prompts=[prompt],
            raw_text=(
                f"[consistency {vote_count}/{self._n_samples}] "
                f"{winner.raw_text}"
            ),
            tokens_used=total_tokens,
            provider_finish_reason=winner.provider_finish_reason,
            token_budget_used=max_budget,
            retries=total_retries,
            call_stats=stats,
            parse_status=winner.parse_status,
            parse_error=winner.parse_error,
        )


def _majority_vote(
    samples: List[StrategyResult],
    *,
    return_type: Any = None,
) -> Tuple[StrategyResult, int]:
    """Select the most frequent recovered, type-conforming value.

    Any or None disables type filtering. Ties select the first encountered
    value. If no sample qualifies, return the first sample with zero votes
    for diagnostics. Type conformity does not establish correctness.
    """
    counted = [
        sample for sample in samples
        if sample.parse_status in (ParseOutcome.PARSED, ParseOutcome.FALLBACK)
        and not validate_output(sample.value, return_type, strict=False)
    ]
    if not counted:
        return samples[0], 0

    groups: List[Tuple[StrategyResult, int]] = []
    for sample in counted:
        for index, (representative, count) in enumerate(groups):
            if _same_value(sample.value, representative.value):
                groups[index] = (representative, count + 1)
                break
        else:
            groups.append((sample, 1))

    # max retains the first group when counts tie.
    return max(groups, key=lambda group: group[1])
