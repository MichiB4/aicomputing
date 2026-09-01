"""
Direct prediction strategy.

The model receives the signature, docstring, and inputs, and directly
predicts the output as a JSON result without any reasoning trace.

Token budget: this strategy resolves think=False unless the request says
otherwise, so get_token_budget() returns 512 for most models — the answer
is a short JSON object. Models classified ALWAYS_THINKS (DeepSeek-R1 and
friends) get 8192 even here, because their reasoning trace cannot be
disabled and would otherwise consume the whole budget. A request-level
max_new_tokens overrides both.

If the model hits the limit the budget doubles and the call retries, up to
5 doublings and never above the 32768 cap in strategies/base.py — so from
512 the sequence tops out at 16384, while a request that starts at 8192
reaches the cap after two doublings.
"""

from __future__ import annotations

from typing import Optional

from ..backends.base import Backend
from ..model_config import get_token_budget
from ..parsing import parse_result
from ..prompting import build_direct_prompt, build_retry_prompt
from ..request import ExecuteRequest
from .base import (CallStats, Strategy, StrategyResult,
                   generate_with_retry, make_parser, _is_strict)


class DirectStrategy(Strategy):
    """Predict the output directly from the spec, no reasoning trace."""

    def __init__(self, include_fewshot: bool = True) -> None:
        self._include_fewshot = include_fewshot

    @property
    def name(self) -> str:
        return "direct"

    def execute(
        self,
        request: ExecuteRequest,
        backend: Backend,
        feedback: Optional[dict] = None,
    ) -> StrategyResult:
        prompt = build_direct_prompt(
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

        think = request.think if request.think is not None else False
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
