"""Route predictions to direct or structured-CoT prompting using heuristics.

The classifier inspects signature/docstring keywords and selected input shapes.
It chooses one existing strategy and delegates prediction to it, forwarding
validation feedback when supplied.

These fixed rules do not establish which strategy is more accurate,
faster, or cheaper for a particular request. Routing does not execute
generated code or expand the runtime's supported input capabilities.

Keyword matching uses whole-word boundaries after lowercasing and treating
underscores as spaces. This supports snake_case identifiers while avoiding
matches inside unrelated words.

The selected strategy is recorded in substrategy and in a raw-text prefix.
The delegated strategy handles generation, parsing, and token-limit retries;
the runtime handles input checks, output validation, and validation retries.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from ..backends.base import Backend
from ..request import ExecuteRequest
from .base import Strategy, StrategyResult
from .cot import CoTStrategy
from .direct import DirectStrategy


# Keywords that route to direct prediction.
_NLP_KEYWORDS = {
    "sentiment", "classify", "categorize", "summarize",
    "translate", "extract", "parse_text", "detect", "score_review",
    "analyze", "describe", "caption", "label", "tag", "topic",
    "emotion", "intent", "language", "tone",
}

# Algorithmic keywords used by the structured-CoT routing rule.
_ALGO_KEYWORDS = {
    "sort", "search", "path", "dijkstra", "bfs", "dfs",
    "traverse", "shortest", "longest", "dynamic", "knapsack",
    "fibonacci", "factorial", "permutation", "combination",
    "graph", "tree", "merge", "partition", "binary_search",
    "topological", "spanning", "optimize",
}

# Mathematical keywords used by the structured-CoT routing rule.
_MATH_KEYWORDS = {
    "integral", "derivative", "solve", "equation", "matrix",
    "determinant", "eigenvalue", "regression",
    "interpolate", "interpolation",
    "numerical", "approximate", "converge", "newton",
    "fourier", "transform", "probability", "distribution",
    "compute", "calculate", "formula",
}


def _compile_keyword_pattern(keywords: set[str]) -> re.Pattern:
    """Compile literal keyword alternatives with word boundaries.

    Normalize underscores to spaces to match the normalization applied
    to request signatures and docstrings.
    """
    escaped = [
        re.escape(keyword.replace("_", " "))
        for keyword in sorted(keywords)
    ]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b")


_NLP_PATTERN = _compile_keyword_pattern(_NLP_KEYWORDS)
_ALGO_PATTERN = _compile_keyword_pattern(_ALGO_KEYWORDS)
_MATH_PATTERN = _compile_keyword_pattern(_MATH_KEYWORDS)


class HybridStrategy(Strategy):
    """Delegate each request to direct or CoT using fixed heuristics."""

    def __init__(self) -> None:
        self._direct = DirectStrategy()
        self._cot = CoTStrategy()

    @property
    def name(self) -> str:
        return "hybrid"

    def cache_state(self) -> dict[str, Any]:
        """Include each delegate's class and configuration in the cache key."""
        state = super().cache_state()
        for attribute in ("_direct", "_cot"):
            delegate = getattr(self, attribute)
            state[attribute] = {
                "strategy_class": (
                    type(delegate).__module__,
                    type(delegate).__qualname__,
                ),
                "strategy_state": delegate.cache_state(),
            }
        return state

    def classify_task(self, request: ExecuteRequest) -> str:
        """Return 'direct' or 'cot' using fixed request heuristics."""
        sig_lower = request.signature.lower()
        doc_lower = request.docstring.lower()
        # Normalize snake_case for whole-word matching.
        combined = (sig_lower + " " + doc_lower).replace("_", " ")

        # NLP keywords take priority over mathematical/algorithmic keywords.
        if _NLP_PATTERN.search(combined):
            return "direct"

        if _MATH_PATTERN.search(combined) or _ALGO_PATTERN.search(combined):
            return "cot"

        # Route selected input shapes to structured-CoT prompting.
        if _has_complex_inputs(request.inputs):
            return "cot"

        # Use direct prediction when no routing rule matches.
        return "direct"

    def execute(
        self,
        request: ExecuteRequest,
        backend: Backend,
        feedback: Optional[dict] = None,
    ) -> StrategyResult:
        strategy_name = self.classify_task(request)

        sub = self._cot if strategy_name == "cot" else self._direct

        result = sub.execute(request, backend, feedback=feedback)
        # Record the selected route in structured metadata and raw text.
        result.substrategy = strategy_name
        result.raw_text = f"[hybrid -> {strategy_name}] " + result.raw_text
        return result


def _has_complex_inputs(inputs: Dict[str, Any]) -> bool:
    """Match the input-shape heuristics used for CoT routing.

    A dictionary input qualifies if an immediate value is a list,
    dictionary, or tuple. A list input qualifies above twenty items.
    """
    for value in inputs.values():
        if isinstance(value, dict) and any(
            isinstance(item, (list, dict, tuple))
            for item in value.values()
        ):
            return True
        if isinstance(value, list) and len(value) > 20:
            return True
    return False
