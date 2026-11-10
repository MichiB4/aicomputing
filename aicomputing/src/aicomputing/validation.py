"""Validate declared input/output types and caller-supplied output predicates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, get_args, get_origin

from .types import ParsedSignature, validate_type_annotation


def validate_nonblank_text(value: object, label: str, *, optional: bool = False) -> None:
    """Require nonblank text without rewriting it; optionally allow None."""
    if value is None and optional:
        return
    if not isinstance(value, str):
        raise TypeError(label + (" must be a string or None" if optional else " must be a string"))
    if not value.strip():
        raise ValueError(label + " must be a nonempty string")


def validate_bool(value: object, label: str) -> None:
    """Require a Boolean without converting other values."""
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a bool")


# Output validation policy.

class ValidationMode(Enum):
    """Output policy: STRICT withholds invalid values; WARN returns diagnostics.

    NONE preserves parsed values without type conversion or output checks.
    Output-validation failures are reported rather than raised.
    Mechanical validation and model self-checking do not establish correctness.
    """

    STRICT = "strict"
    WARN = "warn"
    NONE = "none"

    @classmethod
    def coerce(
        cls,
        value: ValidationMode | str | None,
        default: ValidationMode | None = None,
    ) -> ValidationMode:
        """Resolve an explicit mode or inherit the default, which is STRICT.

        Boolean aliases are not accepted.
        """
        if value is None:
            value = default if default is not None else cls.STRICT
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                raise ValueError(
                    f"unknown validation mode {value!r}; "
                    f"expected one of {[m.value for m in cls]}"
                ) from None
        raise TypeError(f"cannot interpret {value!r} as a ValidationMode")


# Validation errors.

class AITypeError(TypeError):
    """Raised when inputs or outputs violate the declared types."""


class AIConstraintError(ValueError):
    """Represents an output constraint-validation failure."""


# Type-checking helpers.

def check_value_type(value: Any, expected: Any, label: str = "value") -> None:
    """Check supported scalar/container annotations; mismatches raise AITypeError.

    Validate the annotation once, then check container values recursively.
    """
    validate_type_annotation(expected)
    _check_value_type(value, expected, label)


def _check_value_type(value: Any, expected: Any, label: str) -> None:
    """Check a value recursively against an already validated annotation."""
    if expected is Any:
        return

    origin = get_origin(expected)

    # NoneType
    if expected is type(None):
        if value is not None:
            raise AITypeError(
                f"{label}: expected None, got {type(value).__name__}"
            )
        return

    # Inspect generic annotations and validate supported container elements.
    if origin is not None:
        base = origin
        args = tuple(
            type(None) if arg is None else arg for arg in get_args(expected)
        )

        if not isinstance(value, base):
            raise AITypeError(
                f"{label}: expected {base.__name__}[...], "
                f"got {type(value).__name__}"
            )

        if base is list and args:
            elem_type = args[0]
            for i, item in enumerate(value):
                _check_value_type(item, elem_type, label=f"{label}[{i}]")

        elif base is dict and len(args) >= 2:
            k_type, v_type = args[0], args[1]
            for k, v in value.items():
                _check_value_type(k, k_type, label=f"{label}.key({k!r})")
                _check_value_type(v, v_type, label=f"{label}[{k!r}]")

        elif base is set and args:
            elem_type = args[0]
            for item in value:
                _check_value_type(item, elem_type, label=f"{label}.elem")

        elif base is tuple and args:
            if len(args) == 2 and args[1] is Ellipsis:
                for i, item in enumerate(value):
                    _check_value_type(item, args[0], label=f"{label}[{i}]")
            else:
                if len(value) != len(args):
                    raise AITypeError(
                        f"{label}: expected tuple of length {len(args)}, "
                        f"got length {len(value)}"
                    )
                for i, (item, t) in enumerate(zip(value, args)):
                    _check_value_type(item, t, label=f"{label}[{i}]")
        return

    # Ordinary class annotations, including unparameterized containers.
    if isinstance(expected, type):
        # Reject booleans for numeric requirements despite bool inheriting int.
        if expected in (int, float) and isinstance(value, bool):
            raise AITypeError(
                f"{label}: expected {expected.__name__}, got bool"
            )

        # Accept integers for float requirements without converting the value.
        if expected is float and isinstance(value, (int, float)):
            return

        if not isinstance(value, expected):
            raise AITypeError(
                f"{label}: expected {expected.__name__}, "
                f"got {type(value).__name__}"
            )
        return

    # None represents an unspecified requirement.
    return


# Input validation.

def validate_inputs(
    sig: ParsedSignature,
    inputs: Dict[str, Any],
) -> None:
    """Reject invalid argument mappings, names or types before model generation."""
    if not isinstance(inputs, dict):
        raise AITypeError("Inputs must be a dictionary of named arguments")
    declared_names = {p.name for p in sig.params}

    # Check for missing required arguments.
    for p in sig.params:
        if p.name not in inputs and p.default is None:
            raise AITypeError(f"Missing required input: {p.name!r}")

    # Check for unexpected arguments.
    for key in inputs:
        if key not in declared_names:
            raise AITypeError(f"Unexpected input: {key!r} (not in signature)")

    # Type-check each provided argument.
    for p in sig.params:
        if p.name not in inputs:
            continue
        check_value_type(inputs[p.name], p.resolved_type, label=f"input '{p.name}'")


# Output validation.

def validate_output(
    value: Any,
    return_type: Any,
    strict: bool = True,
) -> List[str]:
    """Check output types: strict raises AITypeError; permissive returns warnings.

    Any/None disables checks; type(None) requires a None value.
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


# Constraint validation.

@dataclass
class Constraint:
    """A named output predicate; definition validation never calls it."""

    name: str
    predicate: Callable[[Any], bool]
    message: str = ""

    def __post_init__(self) -> None:
        self.validate_definition()

    def validate_definition(self) -> None:
        """Check declared fields, including changes made after construction."""
        validate_nonblank_text(self.name, "Constraint name")
        if not callable(self.predicate):
            raise TypeError("Constraint predicate must be callable")
        if not isinstance(self.message, str):
            raise TypeError("Constraint message must be a string")

    def check(self, value: Any) -> bool:
        """Run the predicate and require a Boolean result."""
        result = self.predicate(value)
        if not isinstance(result, bool):
            raise TypeError(
                f"Constraint {self.name!r} predicate must return bool, "
                f"got {type(result).__name__}"
            )
        return result


def validate_constraint_definitions(constraints: List[Constraint]) -> None:
    """Validate all declarations without evaluating any output predicate."""
    if not isinstance(constraints, list):
        raise TypeError("constraints must be a list of Constraint objects")
    for index, constraint in enumerate(constraints):
        if not isinstance(constraint, Constraint):
            raise TypeError(f"constraints[{index}] must be a Constraint")
        constraint.validate_definition()


def validate_constraints(
    value: Any,
    constraints: List[Constraint],
    strict: bool = True,
) -> List[str]:
    """Apply predicates: strict raises AIConstraintError; permissive returns failures.

    Definitions are checked before any predicate runs. Predicate exceptions
    propagate unchanged in either mode.
    """
    validate_constraint_definitions(constraints)
    failures: List[str] = []
    for c in constraints:
        ok = c.check(value)
        if not ok:
            msg = c.message or f"Constraint {c.name!r} failed"
            if strict:
                raise AIConstraintError(msg)
            failures.append(msg)
    return failures
