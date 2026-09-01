"""Strategy registry and factory."""

from __future__ import annotations

from typing import Dict

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

# Registry order is the ladder of committed structure — how much intermediate
# representation the model must produce before it is allowed to answer. See
# the prompting module docstring.
_REGISTRY: Dict[str, type] = {
    "direct": DirectStrategy,
    "cot": CoTStrategy,
    "pseudocode": PseudocodeStrategy,
    "contract": ContractStrategy,
    "confidence": ConfidenceStrategy,
    "consistency": SelfConsistencyStrategy,
    "hybrid": HybridStrategy,
}

# Strategies registered and runnable, but not part of the evaluation matrix.
#
# `hybrid` routes between direct and cot with a hand-written keyword
# classifier. It is excluded as an arm for two reasons, neither of them about
# implementation quality:
#
#   1. It contradicts the paradigm. A runtime that inspects a docstring to
#      decide how to execute is a dispatcher carrying a human-authored
#      taxonomy, not a runtime. The classifier also cannot improve as models
#      improve — `confidence` is the same decision handed to the model, which
#      is why that one is an arm and this one is not.
#   2. Its score carries no new information. Because it instantiates the other
#      two strategies and delegates, its per-example result is whatever the
#      routed sub-strategy returned. Given per-example `direct` and `cot`
#      results plus `HybridStrategy.classify_task`, the hybrid score is
#      computable offline — and against the pilot runs the routing was
#      unanimous on three of four benchmarks, so the column largely restates
#      the benchmark axis.
#
# It stays registered so that the routing analysis (keyword routing against a
# perfect-routing oracle upper bound) can be computed, and so it can still be
# run deliberately with an explicit --strategies hybrid.
# `consistency` is excluded for a different reason, and the distinction
# matters: hybrid is excluded because it contradicts the paradigm, whereas
# self-consistency is a sound technique that is simply not one of the five
# treatments under study. It samples the `cot` prompt n times and majority-
# votes, so it varies decoding, not the specification-execution mechanism the
# five arms differ on -- and it costs 5x the calls, which would dominate the
# cost analysis while answering a question about sampling rather than about
# AI computing. It remains fully supported and can be run explicitly with
# --strategies consistency for a separate sampling study.
_NON_ARM = frozenset({"hybrid", "consistency"})


def list_strategies() -> list[str]:
    """
    Canonical list of registered strategy names, in registry order.

    Single source of truth for CLI --strategy/--strategies choices —
    import this instead of hardcoding the name list, so a new strategy
    registered here doesn't need updating in five different places.

    Includes strategies excluded from the evaluation matrix; use list_arms()
    for the matrix default.
    """
    return list(_REGISTRY.keys())


def list_arms() -> list[str]:
    """
    Registered strategies that are arms of the evaluation matrix.

    This is what `scripts/run_evaluation.py` runs by default. It differs from
    list_strategies() only by the exclusions in _NON_ARM, each documented
    there with the reason it is not an arm. Anything excluded remains
    selectable explicitly, so a demoted strategy can still be run on purpose.
    """
    return [name for name in _REGISTRY if name not in _NON_ARM]


def create_strategy(name: str, **kwargs) -> Strategy:
    """
    Create a strategy by name.

    Available strategies (all predict results without generating code):
      - "direct": predict the output directly from the spec
      - "cot": structured chain-of-thought with a state log, then predict
      - "pseudocode": commit to a plain-English procedure, then simulate it
        (see pseudocode.py)
      - "contract": predict, then test the candidate against the docstring
        and revise if it fails (see contract.py)
      - "confidence": model self-assesses, per-call, whether to reason
        first — no external routing logic (see confidence.py)
      - "consistency": samples 5 independent CoT traces and returns the
        majority-voted result (see consistency.py)
      - "hybrid": keyword-classifier routing between direct and cot per-task.
        Registered but not an evaluation arm — see _NON_ARM above.

    >>> strategy = create_strategy("cot")
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy {name!r}. Available: {list(_REGISTRY.keys())}"
        )
    return cls(**kwargs)
