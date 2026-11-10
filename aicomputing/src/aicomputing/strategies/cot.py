"""Request an execution trace with variable states and a predicted output.

The visible trace is a prompt instruction, independent of native reasoning.
It does not establish trace or prediction correctness.
"""

from ..prompting import build_cot_prompt
from .base import PromptStrategy


class CoTStrategy(PromptStrategy):
    """Predict output via structured chain-of-thought with state tracking."""

    name = "cot"
    prompt_builder = staticmethod(build_cot_prompt)
