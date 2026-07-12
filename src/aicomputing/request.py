"""
Core request/result data structures for AI Computing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .validation import Constraint


@dataclass
class ExecuteRequest:
    """
    A request to execute an AI function.

    The programmer declares *what* the function does (signature + docstring),
    provides concrete inputs, and the VirtualRuntime predicts the output.
    """
    signature: str
    docstring: str
    inputs: Dict[str, Any]

    # Type enforcement (mirrors the formal T-AI rule)
    return_type: Optional[Any] = None

    # Optional constraint predicates on the output
    constraints: List[Constraint] = field(default_factory=list)

    # Generation parameters
    temperature: float = 0.0
    top_p: float = 1.0

    # Token budget override. None = auto (model_config.get_token_budget
    # picks 512 for non-thinking, 8192 for thinking models). Set explicitly
    # to force a specific starting budget.
    max_new_tokens: Optional[int] = None

    # Strategy override (None = use runtime default)
    strategy: Optional[str] = None

    # Model override (None = use runtime default)
    model: Optional[str] = None

    # Whether to enforce type-checking strictly (raise on mismatch).
    # None = use the runtime's strict_types setting.
    strict_types: Optional[bool] = None

    # Maximum retries if output fails validation
    max_retries: int = 1

    # Control thinking mode per-call (None = let strategy decide)
    think: Optional[bool] = None


@dataclass
class ExecuteResult:
    """Result of an AI function execution."""
    result: Any
    raw_text: str
    model: str
    strategy: str
    cached: bool = False

    # Validation info
    type_warnings: List[str] = field(default_factory=list)
    constraint_warnings: List[str] = field(default_factory=list)

    # Diagnostics
    retries: int = 0          # validation retries (output failed parsing/validation)
    token_retries: int = 0    # token-budget doublings (model hit length limit)
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
