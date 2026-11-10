"""Request a plain-English procedure, its simulation, and a predicted output.

Both stages share one completion and remain in the response text. The procedure
is neither executed nor frozen separately; simulation adherence is not verified.
"""

from ..prompting import build_pseudocode_prompt
from .base import PromptStrategy


class PseudocodeStrategy(PromptStrategy):
    """Request a plain-English procedure and its simulation on the inputs."""

    name = "pseudocode"
    prompt_builder = staticmethod(build_pseudocode_prompt)
