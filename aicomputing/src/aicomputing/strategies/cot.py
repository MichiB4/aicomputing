"""
Structured Chain-of-Thought (SCoT) strategy.

The model traces Python function execution symbolically — sequential steps,
branches, loops — maintaining an explicit variable-state log at each step,
then outputs the final answer as JSON.

Token budget: this strategy resolves think=True unless the request says
otherwise, so get_token_budget() returns 8192 for every model, including
non-thinking instruct ones — the visible SCoT trace needs the room even
when the model has no hidden reasoning trace. Passing think=False on the
request drops a non-thinking model to 512; a request-level max_new_tokens
overrides both.

If the model hits the limit the budget doubles and the call retries, up to
5 doublings and never above the 32768 cap in strategies/base.py — so from
8192 the sequence is 8192→16384→32768.
"""

from __future__ import annotations

from typing import Optional

from ..backends.base import Backend
from ..model_config import get_token_budget
from ..parsing import parse_result
from ..prompting import build_cot_prompt, build_retry_prompt
from ..request import ExecuteRequest
from .base import (CallStats, Strategy, StrategyResult,
                   generate_with_retry, make_parser, _is_strict)


class CoTStrategy(Strategy):
    """Predict output via structured chain-of-thought with state tracking."""

    def __init__(self, include_fewshot: bool = True) -> None:
        self._include_fewshot = include_fewshot

    @property
    def name(self) -> str:
        return "cot"

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
