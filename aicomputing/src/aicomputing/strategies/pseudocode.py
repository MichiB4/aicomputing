"""
Pseudocode-then-simulate strategy.

The model writes the procedure the docstring describes as numbered plain-English
steps, then simulates that procedure on the concrete inputs and reports what it
produced.

Why this is not `cot`. SCoT traces execution directly from the specification,
so nothing in the prompt stops the model from settling on a plausible-looking
answer first and narrating a trace that agrees with it — the failure mode CoRE
(Gao et al., 2026) documents as *superficial execution*, where output accuracy
far exceeds the fidelity of the intermediate states claimed along the way. Here
the procedure is committed in stage 1 and the value is whatever stage 2 gets by
executing it, which makes the two stages separately inspectable in the trace.

Why this is not the baseline. The procedure is deliberately non-executable: the
prompt asks for plain English and forbids language syntax. Adapted from the
pseudocode-simulation path of DuET (Han et al., 2026), whose other path compiles
real code and runs it in a real interpreter. That second path is what
`benchmarks/baselines/code_gen_baseline.py` measures and is explicitly not AI
computing — the moment an interpreter produces the value, the model has stopped
being the runtime.

Single call. Both stages are produced in one completion, which keeps the cost
comparable to `cot` and keeps the strategy a single consultation of the oracle,
matching the shape of the E-AI evaluation rule.

Token budget: resolves think=True unless the request says otherwise, so
get_token_budget() returns 8192 — two stages need more room than one. Passing
think=False on the request drops a non-thinking model to 512; a request-level
max_new_tokens overrides both.
"""

from __future__ import annotations

from typing import Optional

from ..backends.base import Backend
from ..model_config import get_token_budget
from ..parsing import parse_result
from ..prompting import build_pseudocode_prompt, build_retry_prompt
from ..request import ExecuteRequest
from .base import (CallStats, Strategy, StrategyResult,
                   generate_with_retry, make_parser, _is_strict)


class PseudocodeStrategy(Strategy):
    """Commit to a plain-English procedure, then simulate it on the inputs."""

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
