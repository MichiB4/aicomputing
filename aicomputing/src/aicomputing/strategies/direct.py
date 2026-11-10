"""Predict outputs directly from specifications and inputs."""

from ..prompting import build_direct_prompt
from .base import PromptStrategy


class DirectStrategy(PromptStrategy):
    """Predict an output using the direct prompt, without a separate check stage."""

    name = "direct"
    prompt_builder = staticmethod(build_direct_prompt)
