"""Let the model answer immediately when confident, otherwise trace first.

Both styles use one prompt. Visible traces remain in the response; confidence,
calibration, and the chosen response path are not measured separately.
"""

from ..prompting import build_confidence_prompt
from .base import PromptStrategy


class ConfidenceStrategy(PromptStrategy):
    """Ask the model to choose whether to trace before answering."""

    name = "confidence"
    prompt_builder = staticmethod(build_confidence_prompt)
