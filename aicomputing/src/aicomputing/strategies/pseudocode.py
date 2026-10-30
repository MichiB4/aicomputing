"""Request a plain-English procedure, its simulation, and a predicted output.

Both stages share one completion and remain in the response text. The procedure
is neither executed nor frozen separately; simulation adherence is not verified.
"""

from __future__ import annotations

from typing import Optional

from ..backends.base import Backend
from ..prompting import build_pseudocode_prompt
from ..request import ExecuteRequest
from .base import Strategy, StrategyResult, _execute_prompt


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
            answer_format=request.answer_format,
        )
        return _execute_prompt(request, backend, prompt, feedback)
