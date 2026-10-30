"""Let the model answer immediately when confident, otherwise trace first.

Both styles use one prompt. Visible traces remain in the response; confidence,
calibration, and the chosen response path are not measured separately.
"""

from __future__ import annotations

from typing import Optional

from ..backends.base import Backend
from ..prompting import build_confidence_prompt
from ..request import ExecuteRequest
from .base import Strategy, StrategyResult, _execute_prompt


class ConfidenceStrategy(Strategy):
    """Ask the model to choose whether to trace before answering."""

    def __init__(self, include_fewshot: bool = True) -> None:
        self._include_fewshot = include_fewshot

    @property
    def name(self) -> str:
        return "confidence"

    def execute(
        self,
        request: ExecuteRequest,
        backend: Backend,
        feedback: Optional[dict] = None,
    ) -> StrategyResult:
        prompt = build_confidence_prompt(
            signature=request.signature,
            docstring=request.docstring,
            inputs=request.inputs,
            include_fewshot=self._include_fewshot,
            answer_format=request.answer_format,
        )
        return _execute_prompt(request, backend, prompt, feedback)
