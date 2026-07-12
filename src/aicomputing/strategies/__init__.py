"""Strategy registry and factory."""

from __future__ import annotations

from typing import Dict

from .base import Strategy, StrategyResult
from .direct import DirectStrategy
from .cot import CoTStrategy
from .hybrid import HybridStrategy
from .confidence import ConfidenceStrategy

__all__ = [
    "Strategy",
    "StrategyResult",
    "DirectStrategy",
    "CoTStrategy",
    "HybridStrategy",
    "ConfidenceStrategy",
    "create_strategy",
]

_REGISTRY: Dict[str, type] = {
    "direct": DirectStrategy,
    "cot": CoTStrategy,
    "hybrid": HybridStrategy,
    "confidence": ConfidenceStrategy,
}


def create_strategy(name: str, **kwargs) -> Strategy:
    """
    Create a strategy by name.

    Available strategies (all predict results without generating code):
      - "direct": predict the output directly from the spec
      - "cot": chain-of-thought reasoning, then predict
      - "hybrid": keyword-classifier routing between direct and cot per-task
      - "confidence": model self-assesses, per-call, whether to reason
        first — no external routing logic (see confidence.py)

    >>> strategy = create_strategy("cot")
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy {name!r}. Available: {list(_REGISTRY.keys())}"
        )
    return cls(**kwargs)
