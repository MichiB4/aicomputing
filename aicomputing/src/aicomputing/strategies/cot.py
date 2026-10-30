"""Request an execution trace with variable states and a predicted output.

The visible trace is a prompt instruction, independent of native reasoning.
It does not establish trace or prediction correctness.
"""

from __future__ import annotations

from typing import Optional

from ..backends.base import Backend
from ..prompting import build_cot_prompt
from ..request import ExecuteRequest
from .base import Strategy, StrategyResult, _execute_prompt


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
            answer_format=request.answer_format,
        )
        return _execute_prompt(request, backend, prompt, feedback)
