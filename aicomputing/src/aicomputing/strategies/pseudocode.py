"""Predict outputs by requesting a procedure followed by its simulation.

The prompt asks the model to describe a procedure as numbered plain-English
steps, simulate those steps on the concrete inputs, and return the predicted
answer in a result object. No interpreter executes the generated procedure.

Both stages are requested in one completion. The strategy does not freeze
the procedure separately or verify that the simulation follows it.
The response text retains both stages when the model includes them.

Native reasoning is configured by the backend rather than selected by
this strategy. The requested procedure and simulation are prompt instructions.
The initial output-token budget comes from get_token_budget unless
request.max_new_tokens supplies one.

Token-limit retries can make up to five additional calls, doubling the
budget without a package-level cap. The runtime may also repeat the strategy
after parsing or validation failures, according to request.max_retries.
"""

from __future__ import annotations

from typing import Optional

from ..backends.base import Backend
from ..model_config import get_token_budget
from ..prompting import build_pseudocode_prompt, build_retry_prompt
from ..request import ExecuteRequest
from .base import (
    CallStats,
    Strategy,
    StrategyResult,
    _is_strict,
    generate_with_retry,
    make_parser,
)


class PseudocodeStrategy(Strategy):
    """Request a plain-English procedure and its simulation on the inputs."""

    def __init__(self, include_fewshot: bool = True) -> None:
        self._include_fewshot = include_fewshot

    @property
    def name(self) -> str:
        return "pseudocode"

    def execute(
        self,
        request: ExecuteRequest,
        backend: Backend,
        feedback: Optional[dict] = None,
    ) -> StrategyResult:
        prompt = build_pseudocode_prompt(
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

        # Preserve the existing default allowance; this does not set native reasoning.
        initial_budget = request.max_new_tokens or get_token_budget(
            request.model or backend.default_model, extended=True,
        )

        def call(max_new_tokens: int):
            return backend.generate(
                prompt,
                model=request.model,
                temperature=request.temperature,
                top_p=request.top_p,
                max_new_tokens=max_new_tokens,
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
