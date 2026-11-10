"""Strategy registry and factory."""

from __future__ import annotations

from ..validation import validate_nonblank_text
from .base import Strategy, StrategyResult
from .direct import DirectStrategy
from .cot import CoTStrategy
from .pseudocode import PseudocodeStrategy
from .contract import ContractStrategy
from .hybrid import HybridStrategy
from .confidence import ConfidenceStrategy
from .consistency import SelfConsistencyStrategy
from .code_execute import CodeExecuteStrategy, CodeExecutionError

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
    "CodeExecuteStrategy",
    "CodeExecutionError",
    "create_strategy",
    "list_strategies",
]

# Preserve this order when listing strategies.
_REGISTRY: dict[str, type[Strategy]] = {
    "direct": DirectStrategy,
    "cot": CoTStrategy,
    "pseudocode": PseudocodeStrategy,
    "contract": ContractStrategy,
    "confidence": ConfidenceStrategy,
    "consistency": SelfConsistencyStrategy,
    "hybrid": HybridStrategy,
    "code_execute": CodeExecuteStrategy,
}

def list_strategies() -> list[str]:
    """Return all registered strategy names in registry order."""
    return list(_REGISTRY)


def create_strategy(name: str, **kwargs) -> Strategy:
    """Create a registered strategy using its exact, case-sensitive name.

    Forward keyword arguments to the strategy constructor.
    Unknown names raise ValueError; constructor errors propagate unchanged.

    >>> strategy = create_strategy("cot")
    """
    validate_nonblank_text(name, "strategy")
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy {name!r}. Available: {list_strategies()}"
        )
    return cls(**kwargs)
