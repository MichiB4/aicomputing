"""Self-consistency prediction using repeated structured-CoT samples.

Selects the most frequent recovered value matching the expected return type,
with ties resolved by first occurrence. Any leaves voting unrestricted.
None participates when its type is allowed; failed recovery does not vote.
The selected sample supplies answer text and parsing metadata, while
usage is accumulated across samples.
Samples use the request's temperature; at 0.0 a deterministic provider returns
identical samples, so use a positive temperature for meaningful voting.
Native reasoning is configured by the backend rather than selected by
this strategy.
Samples have no application output-token cap unless the request supplies one.
Provider output and context limits still apply.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from ..backends.base import Backend
from ..status import ParseStatus
from ..prompting import build_cot_prompt
from ..request import ExecuteRequest
from ..validation import validate_bool, validate_output
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


class SelfConsistencyStrategy(Strategy):
    """Sample `n_samples` CoT traces and return the most frequent value."""

    def __init__(self, n_samples: int = 5, include_fewshot: bool = True) -> None:
        if type(n_samples) is not int:
            raise TypeError("n_samples must be an integer")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        validate_bool(include_fewshot, "include_fewshot")
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
        prompt = _with_feedback(build_cot_prompt(
            signature=request.signature,
            docstring=request.docstring,
            inputs=request.inputs,
            include_fewshot=self._include_fewshot,
        ), feedback)
        call = _backend_call(request, backend, prompt)
        return_type = _output_type(request)
        parse = make_parser(return_type)

        # Accumulate usage across all planned samples, including truncated ones.
        stats = CallStats()
        samples: List[StrategyResult] = []
        for _ in range(self._n_samples):
            outcome, response = _generate_once(
                call=call,
                parse=parse,
                stats=stats,
            )
            samples.append(StrategyResult(
                value=outcome.value,
                raw_text=response.text,
                feedback_text=response.parsing_text,
                provider_finish_reason=response.provider_finish_reason,
                parse_status=outcome.status,
                parse_error=outcome.error,
            ))

        winner, vote_count = _select_most_frequent(
            samples, return_type=return_type,
        )

        return StrategyResult(
            value=winner.value,
            # All samples use this prompt; record it once.
            prompts=[prompt],
            raw_text=(
                f"[consistency {vote_count}/{self._n_samples}] "
                f"{winner.raw_text}"
            ),
            feedback_text=(
                f"[consistency {vote_count}/{self._n_samples}] "
                f"{winner.feedback_text}"
            ),
            provider_finish_reason=winner.provider_finish_reason,
            call_stats=stats,
            parse_status=winner.parse_status,
            parse_error=winner.parse_error,
        )


def _select_most_frequent(
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
        if sample.parse_status in (ParseStatus.PARSED, ParseStatus.FALLBACK)
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
