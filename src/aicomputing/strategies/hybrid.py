"""
Hybrid / adaptive strategy.

Routes each AI function call to the most appropriate prediction strategy
based on heuristics about the task type. All strategies predict the result
directly from the specification — no code is generated or executed.

Current heuristics (rule-based; can be replaced with a learned classifier):
- NLP/perception tasks (sentiment, classification, ...) -> direct
- Simple operations (arithmetic, string ops, lookups) -> direct
- Multimodal tasks (Image, Audio, Video inputs/outputs) -> direct
- Algorithmic tasks (graph, sorting, DP, complex types) -> cot
- Mathematical/numerical tasks (formulas, computation) -> cot

PROVISIONAL: this is a candidate answer to open question #1 in CLAUDE.md
("universal vs. task-specific prompting strategy"), not a validated one —
it is not empirically benchmarked against plain `direct` or `cot`, and has
no dedicated test coverage beyond classify_task() unit tests. Treat routing
decisions as a heuristic prior, not a proven policy.

Keyword matching is whole-word only (regex \\b...\\b, underscores treated
as word separators) specifically to avoid substring false positives, e.g.
the old naive `"tag" in text` check used to fire on "postage"/"voltage",
and `"path" in text` fired on "empathy"/"pathology". Don't revert to plain
`kw in text` substring checks.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from ..backends.base import Backend
from ..request import ExecuteRequest
from ..types import ParsedSignature, parse_signature, Image, Audio, Video, Text, MediaObject
from .base import Strategy, StrategyResult
from .direct import DirectStrategy
from .cot import CoTStrategy


# Keywords suggesting NLP/perception tasks -> direct (pattern matching, no steps)
_NLP_KEYWORDS = {
    "sentiment", "classify", "categorize", "summarize",
    "translate", "extract", "parse_text", "detect", "score_review",
    "analyze", "describe", "caption", "label", "tag", "topic",
    "emotion", "intent", "language", "tone",
}

# Keywords suggesting algorithmic tasks -> cot (multi-step reasoning helps)
_ALGO_KEYWORDS = {
    "sort", "search", "path", "dijkstra", "bfs", "dfs",
    "traverse", "shortest", "longest", "dynamic", "knapsack",
    "fibonacci", "factorial", "permutation", "combination",
    "graph", "tree", "merge", "partition", "binary_search",
    "topological", "spanning", "optimize",
}

# Keywords suggesting mathematical/numerical tasks -> cot (step-by-step helps)
_MATH_KEYWORDS = {
    "integral", "derivative", "solve", "equation", "matrix",
    "determinant", "eigenvalue", "regression", "interpolat",
    "numerical", "approximate", "converge", "newton",
    "fourier", "transform", "probability", "distribution",
    "compute", "calculate", "formula",
}


def _compile_keyword_pattern(keywords: set[str]) -> re.Pattern:
    """
    Compile a whole-word regex for a keyword set.

    Underscores are treated as word separators (so "sort_items" and
    "binary_search" line up with the "_"->" " normalisation applied to the
    signature/docstring text in classify_task), and matching is anchored
    with \\b on both sides so keywords can't fire as a substring of an
    unrelated word.
    """
    escaped = [re.escape(kw.replace("_", " ")) for kw in sorted(keywords, key=len, reverse=True)]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b")


_NLP_PATTERN = _compile_keyword_pattern(_NLP_KEYWORDS)
_ALGO_PATTERN = _compile_keyword_pattern(_ALGO_KEYWORDS)
_MATH_PATTERN = _compile_keyword_pattern(_MATH_KEYWORDS)


class HybridStrategy(Strategy):
    """
    Adaptive strategy that picks direct or cot per-task.

    Both sub-strategies predict results purely from the specification
    (signature + docstring + inputs). The difference is whether the
    model reasons step-by-step first (cot) or answers immediately
    (direct).

    Rationale:
    - Direct is faster and cheaper. It works well for pattern-matching
      tasks (NLP, perception) and simple operations where the model
      can answer from training knowledge alone.
    - CoT is slower but more accurate on multi-step reasoning tasks
      (algorithms, math, complex logic) where intermediate steps
      reduce error accumulation.
    """

    def __init__(self) -> None:
        self._direct = DirectStrategy()
        self._cot = CoTStrategy()

    @property
    def name(self) -> str:
        return "hybrid"

    def classify_task(self, request: ExecuteRequest) -> str:
        """
        Classify a task and return the recommended strategy name.

        Returns one of: 'direct', 'cot'.
        """
        sig_lower = request.signature.lower()
        doc_lower = request.docstring.lower()
        # Normalise "_" to " " so snake_case identifiers (e.g. "sort_items")
        # line up with whole-word keyword matching instead of being glued
        # into one unmatchable token.
        combined = (sig_lower + " " + doc_lower).replace("_", " ")

        # Parse signature for structural hints
        try:
            parsed = parse_signature(request.signature)
        except ValueError:
            parsed = None

        # Multimodal types -> direct (the model's perception IS the impl)
        if parsed and _has_media_types(parsed):
            return "direct"

        # NLP/perception keywords -> direct
        if _NLP_PATTERN.search(combined):
            return "direct"

        # Mathematical keywords -> cot (step-by-step reasoning)
        if _MATH_PATTERN.search(combined):
            return "cot"

        # Algorithmic keywords -> cot
        if _ALGO_PATTERN.search(combined):
            return "cot"

        # Complex nested inputs (graphs, matrices) -> cot
        if _has_complex_inputs(request.inputs):
            return "cot"

        # Float return with computation-like docstring -> cot
        if parsed and parsed.return_type is float:
            return "cot"

        # Default: direct (fastest, good enough for simple tasks)
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
        # Tag the result so callers know which sub-strategy was used
        result.raw_text = f"[hybrid -> {strategy_name}] " + result.raw_text
        return result


def _has_media_types(parsed: ParsedSignature) -> bool:
    """Check if any parameter or return type is a media type."""
    media = (Image, Audio, Video, Text, MediaObject)
    for p in parsed.params:
        if isinstance(p.resolved_type, type) and issubclass(p.resolved_type, media):
            return True
    if isinstance(parsed.return_type, type) and issubclass(parsed.return_type, media):
        return True
    return False


def _has_complex_inputs(inputs: Dict[str, Any]) -> bool:
    """Heuristic: inputs with nested dicts/lists suggest algorithmic tasks."""
    for v in inputs.values():
        if isinstance(v, dict) and any(isinstance(vv, (list, dict, tuple)) for vv in v.values()):
            return True
        if isinstance(v, list) and len(v) > 20:
            return True
    return False
