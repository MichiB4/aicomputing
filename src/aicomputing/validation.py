"""
Runtime type-checking and constraint validation for AI Computing.

Implements the typing rule T-AI from the paper: checks that inputs match
the declared parameter types and that the model's output matches the
declared return type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, get_args, get_origin

from .types import MediaObject, ParsedSignature


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class AITypeError(TypeError):
    """Raised when inputs or outputs violate the declared types."""


class AIConstraintError(ValueError):
    """Raised when a constraint predicate fails on the model output."""


# ---------------------------------------------------------------------------
# Type-checking helpers
# ---------------------------------------------------------------------------

def _is_media_type(t: Any) -> bool:
    try:
        return t is not Any and isinstance(t, type) and issubclass(t, MediaObject)
    except TypeError:
        return False


def check_value_type(value: Any, expected: Any, label: str = "value") -> None:
    """
    Check that *value* conforms to *expected* type.

    Supports: int, float, str, bool, list, dict, tuple, set,
    generic forms (list[int], dict[str, int]), None, Any,
    and multimodal types (Image, Audio, Video, Text, MediaObject).
    """
    if expected is Any:
        return  # anything goes

    origin = get_origin(expected)

    # NoneType
    if expected is type(None):
        if value is not None:
            raise AITypeError(f"{label}: expected None, got {type(value).__name__}")
        return

    # Media types
    if _is_media_type(expected):
        if not isinstance(value, expected):
            raise AITypeError(
                f"{label}: expected {expected.__name__}, got {type(value).__name__}"
            )
        return

    # Generic: list[X], dict[K,V], tuple[...], set[X]
    if origin is not None:
        base = origin
        args = get_args(expected)

        if not isinstance(value, base):
            raise AITypeError(
                f"{label}: expected {base.__name__}[...], got {type(value).__name__}"
            )

        if base is list and args:
            elem_type = args[0]
            for i, item in enumerate(value):
                check_value_type(item, elem_type, label=f"{label}[{i}]")

        elif base is dict and len(args) >= 2:
            k_type, v_type = args[0], args[1]
            for k, v in value.items():
                check_value_type(k, k_type, label=f"{label}.key({k!r})")
                check_value_type(v, v_type, label=f"{label}[{k!r}]")

        elif base is set and args:
            elem_type = args[0]
            for item in value:
                check_value_type(item, elem_type, label=f"{label}.elem")

        elif base is tuple and args:
            if len(args) == 2 and args[1] is Ellipsis:
                for i, item in enumerate(value):
                    check_value_type(item, args[0], label=f"{label}[{i}]")
            else:
                for i, (item, t) in enumerate(zip(value, args)):
                    check_value_type(item, t, label=f"{label}[{i}]")
        return

    # Plain types: int, float, str, bool
    if isinstance(expected, type):
        # Allow int where float is expected (standard Python widening)
        if expected is float and isinstance(value, (int, float)):
            return
        if not isinstance(value, expected):
            raise AITypeError(
                f"{label}: expected {expected.__name__}, got {type(value).__name__}"
            )
        return

    # Fallback: don't raise on things we can't check
    return


# ---------------------------------------------------------------------------
# Input validation (T-AI rule: Γ ⊢ e : T_in)
# ---------------------------------------------------------------------------

def validate_inputs(
    sig: ParsedSignature,
    inputs: Dict[str, Any],
    strict: bool = True,
) -> List[str]:
    """
    Validate that *inputs* match the parameter types declared in *sig*.

    Returns a list of warning strings. Raises AITypeError in strict mode
    on the first mismatch.
    """
    warnings: List[str] = []
    declared_names = {p.name for p in sig.params}

    # Check for missing required params
    for p in sig.params:
        if p.name not in inputs and p.default is None:
            msg = f"Missing required input: {p.name!r}"
            if strict:
                raise AITypeError(msg)
            warnings.append(msg)

    # Check for unexpected params
    for key in inputs:
        if key not in declared_names:
            msg = f"Unexpected input: {key!r} (not in signature)"
            if strict:
                raise AITypeError(msg)
            warnings.append(msg)

    # Type-check each provided input
    for p in sig.params:
        if p.name not in inputs:
            continue
        if p.resolved_type is Any:
            continue
        try:
            check_value_type(inputs[p.name], p.resolved_type, label=f"input '{p.name}'")
        except AITypeError as e:
            if strict:
                raise
            warnings.append(str(e))

    return warnings


# ---------------------------------------------------------------------------
# Output validation (T-AI rule: result has type T_out)
# ---------------------------------------------------------------------------

def validate_output(
    value: Any,
    return_type: Any,
    strict: bool = True,
) -> List[str]:
    """
    Validate that the model's output matches the declared return type.

    Returns a list of warning strings. Raises AITypeError in strict mode.
    """
    if return_type is Any or return_type is None:
        return []
    try:
        check_value_type(value, return_type, label="return value")
        return []
    except AITypeError as e:
        if strict:
            raise
        return [str(e)]


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------

@dataclass
class Constraint:
    """A named predicate that the output must satisfy."""
    name: str
    predicate: Callable[[Any], bool]
    message: str = ""

    def check(self, value: Any) -> bool:
        return self.predicate(value)


def validate_constraints(
    value: Any,
    constraints: List[Constraint],
    strict: bool = True,
) -> List[str]:
    """
    Run all constraint predicates against *value*.

    Returns list of failure messages. Raises AIConstraintError in strict mode.
    """
    failures: List[str] = []
    for c in constraints:
        try:
            ok = c.check(value)
        except Exception as exc:
            ok = False
            failures.append(f"Constraint {c.name!r} raised {exc}")
            continue
        if not ok:
            msg = c.message or f"Constraint {c.name!r} failed"
            if strict:
                raise AIConstraintError(msg)
            failures.append(msg)
    return failures
