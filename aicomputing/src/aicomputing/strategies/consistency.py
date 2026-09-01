"""
Self-consistency strategy (Wang et al., "Self-Consistency Improves Chain of
Thought Reasoning in Language Models", ICLR 2023).

Samples `n_samples` independent structured-CoT traces at a non-zero
temperature and returns the majority-voted parsed result. Unlike `hybrid`
and `confidence`, this is a well-established, validated technique from the
literature rather than a project-specific heuristic — the tradeoff is
well understood: it costs `n_samples`x the inference calls and latency of
a single `cot` call in exchange for reduced variance on tasks where the
model's reasoning is inconsistent from sample to sample.

Each sample reuses the same structured CoT prompt as the `cot` strategy
so its trace format is directly comparable. `tokens_used` and the
token-budget-retry count reported on the result are summed across all
`n_samples` independent calls, not just the winning sample.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, List, Optional, Tuple

from ..backends.base import Backend, GenerateResponse
from ..model_config import get_token_budget
from ..parsing import parse_result
from ..prompting import build_cot_prompt, build_retry_prompt
from ..request import ExecuteRequest
from .base import (CallStats, Strategy, StrategyResult,
                   generate_with_retry, make_parser, _is_strict)

# Self-consistency needs sampling diversity — greedy decoding (temperature=0)
# would just produce n identical votes. Used only when the caller hasn't
# requested a specific non-zero temperature.
_DEFAULT_SAMPLE_TEMPERATURE = 0.7


class SelfConsistencyStrategy(Strategy):
    """Sample `n_samples` CoT traces and return the majority-voted result."""

    def __init__(self, n_samples: int = 5, include_fewshot: bool = True) -> None:
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

        think = request.think if request.think is not None else True
        initial_budget = request.max_new_tokens or get_token_budget(
            request.model or backend.default_model, think,
        )
        temperature = (
            request.temperature if request.temperature > 0
            else _DEFAULT_SAMPLE_TEMPERATURE
        )

        psink: dict = {}
        parse = make_parser(request.return_type, _is_strict(request), psink)

        def call(max_new_tokens: int):
            return backend.generate(
                prompt,
                model=request.model,
                temperature=temperature,
                top_p=request.top_p,
                max_new_tokens=max_new_tokens,
                think=think,
            )

        # Shared across every sample: the defining cost of this strategy is
        # that it pays n_samples times, and reporting one sample would erase it.
        stats = CallStats()
        samples: List[Tuple[Any, GenerateResponse, int]] = []
        budgets: List[int] = []
        for _ in range(self._n_samples):
            value, response, retries, final_budget = generate_with_retry(
                call=call,
                parse=parse,
                initial_budget=initial_budget,
                model_name=request.model or backend.default_model,
                stats=stats,
            )
            samples.append((value, response, retries))
            budgets.append(final_budget)

        best_value, winning_response, vote_count = _majority_vote(samples)
        total_tokens = sum(r.tokens_used or 0 for _, r, _ in samples) or None
        total_retries = sum(retries for _, _, retries in samples)
        # Samples share a starting budget but may double independently, so
        # there is no single budget for the call. Report the largest ceiling
        # requested by any sample — the figure that matters against a
        # provider's per-request token cap. Consistent with tokens_used and
        # retries, which aggregate across samples rather than describing the
        # winning one.
        max_budget = max(budgets) if budgets else None

        return StrategyResult(
            value=best_value,
            # One prompt, sampled n_samples times -- recorded once,
            # because the samples differ by temperature, not by text.
            prompts=[prompt],
            raw_text=(
                f"[consistency {vote_count}/{self._n_samples}] "
                f"{winning_response.text}"
            ),
            tokens_used=total_tokens,
            provider_finish_reason=winning_response.provider_finish_reason,
            token_budget_used=max_budget,
            retries=total_retries,
            call_stats=stats,
            parse_status=psink.get("parse_status"),
            parse_error=psink.get("parse_error"),
        )


def _majority_vote(
    samples: List[Tuple[Any, GenerateResponse, int]],
) -> Tuple[Any, GenerateResponse, int]:
    """
    Return (value, winning_response, vote_count) for the most common parsed
    value among the samples. Failed parses (None) are excluded from the
    vote unless every sample failed, in which case the first failure is
    returned so the caller still gets a raw_text to inspect.
    """
    def vote_key(value: Any) -> str:
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except TypeError:
            return repr(value)

    counted = [(v, r) for v, r, _ in samples if v is not None]
    if not counted:
        value, response, _ = samples[0]
        return value, response, 0

    counts = Counter(vote_key(v) for v, _ in counted)
    winning_key, vote_count = counts.most_common(1)[0]

    for value, response in counted:
        if vote_key(value) == winning_key:
            return value, response, vote_count
    raise AssertionError("unreachable: winning_key must match a counted sample")
