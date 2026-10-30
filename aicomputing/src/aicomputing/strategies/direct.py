"""Predict outputs directly from specifications and inputs."""

from __future__ import annotations

from typing import Optional

from ..backends.base import Backend
from ..prompting import build_direct_prompt
from ..request import ExecuteRequest
from .base import Strategy, StrategyResult, _execute_prompt


class DirectStrategy(Strategy):
    """Predict an output using the direct prompt, without a separate check stage."""

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
            answer_format=request.answer_format,
        )
        return _execute_prompt(request, backend, prompt, feedback)
