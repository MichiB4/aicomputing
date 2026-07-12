"""
Structured Chain-of-Thought (SCoT) strategy.

The model traces Python function execution symbolically — sequential steps,
branches, loops — maintaining an explicit variable-state log at each step,
then outputs the final answer as JSON.

Token budget: 8192 for thinking models (reasoning trace needs space),
512 for non-thinking instruct models (SCoT trace + JSON fits).
If the model hits the limit, the budget doubles and the call retries
automatically — up to 5 doublings.
"""

from __future__ import annotations

from typing import Optional

from ..backends.base import Backend
from ..model_config import get_token_budget
from ..parsing import parse_result
from ..prompting import build_cot_prompt, build_retry_prompt
from ..request import ExecuteRequest
from .base import Strategy, StrategyResult, generate_with_retry


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

        def parse(text: str):
            return parse_result(text, return_type=request.return_type)

        value, response, retries = generate_with_retry(
            call=call,
            parse=parse,
            initial_budget=initial_budget,
            model_name=request.model or backend.default_model,
        )

        return StrategyResult(
            value=value,
            raw_text=response.text,
            tokens_used=response.tokens_used,
            retries=retries,
        )
