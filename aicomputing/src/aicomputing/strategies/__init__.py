"""Strategy registry and factory."""

from __future__ import annotations

from .base import Strategy, StrategyResult
from .direct import DirectStrategy
from .cot import CoTStrategy
from .pseudocode import PseudocodeStrategy
from .contract import ContractStrategy
from .hybrid import HybridStrategy
from .confidence import ConfidenceStrategy
from .consistency import SelfConsistencyStrategy

__all__ = [
    "Strategy",
    "StrategyResult",
    "DirectStrategy",
    "CoTStrategy",
    "PseudocodeStrategy",
    "ContractStrategy",
    "HybridStrategy",
    "ConfidenceStrategy",
    "SelfConsistencyStrategy",
    "create_strategy",
    "list_strategies",
    "list_arms",
]

# Preserve this order when listing strategies and default evaluation arms.
_REGISTRY: dict[str, type[Strategy]] = {
    "direct": DirectStrategy,
    "cot": CoTStrategy,
    "pseudocode": PseudocodeStrategy,
    "contract": ContractStrategy,
    "confidence": ConfidenceStrategy,
    "consistency": SelfConsistencyStrategy,
    "hybrid": HybridStrategy,
}

# Registered strategies excluded from the default evaluation subset
# returned by list_arms(). Both remain available through create_strategy()
# and are included in list_strategies().
#
# Hybrid routes requests to direct or CoT using fixed heuristics.
# Consistency makes repeated CoT predictions and votes among eligible answers.
#
# These exclusions define experiment scope, not runtime support.
_NON_ARM = frozenset({"hybrid", "consistency"})


def list_strategies() -> list[str]:
    """Return all registered strategy names in registry order.

    Includes strategies excluded from the default evaluation subset.
    """
    return list(_REGISTRY)


def list_arms() -> list[str]:
    """Return default evaluation strategy names in registry order.

    Exclude names in _NON_ARM. Excluded strategies remain available
    through create_strategy() and are included in list_strategies().
    """
    return [name for name in _REGISTRY if name not in _NON_ARM]


def create_strategy(name: str, **kwargs) -> Strategy:
    """Create a registered strategy using its exact, case-sensitive name.

    Forward keyword arguments to the strategy constructor.
    Unknown names raise ValueError; constructor errors propagate unchanged.

    >>> strategy = create_strategy("cot")
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy {name!r}. Available: {list_strategies()}"
        )
    return cls(**kwargs)
