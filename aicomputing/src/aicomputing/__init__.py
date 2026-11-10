"""
AI Computing — LLM-based predictive execution as a programming paradigm.

Quick start::

    from aicomputing import VirtualRuntime, ExecuteRequest

    with VirtualRuntime(backend="ollama", model="qwen3:4b") as vr:
        result = vr.execute(ExecuteRequest(
            signature="sort(arr: list[int]) -> list[int]",
            docstring="Return a sorted copy of the array.",
            inputs={"arr": [5, 3, 1, 4]},
        ))
    print(result.result)  # [1, 3, 4, 5]
"""

from .version import __version__
from .request import ExecuteRequest, ExecuteResult
from .status import ParseStatus, ValidationStatus
from .runtime import VirtualRuntime
from .validation import Constraint, AITypeError, AIConstraintError, ValidationMode
from .backends.base import ToolUseDetected
from .types import parse_signature, ParsedSignature
from .backends import create_backend, Backend
from .strategies import create_strategy, Strategy, CodeExecutionError
from .trace import Tracer

__all__ = [
    "__version__",
    # Core
    "VirtualRuntime",
    "ExecuteRequest",
    "ExecuteResult",
    "ParseStatus",
    "ValidationStatus",
    # Validation
    "Constraint",
    "ValidationMode",
    "AITypeError",
    "ToolUseDetected",
    "CodeExecutionError",
    "AIConstraintError",
    # Types
    "parse_signature",
    "ParsedSignature",
    # Factories
    "create_backend",
    "create_strategy",
    "Backend",
    "Strategy",
    # Tracing
    "Tracer",
]
