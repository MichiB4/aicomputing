"""
Runtime type-checking and constraint validation for AI Computing.

Checks inputs against declared parameter types, outputs against expected
return types, and outputs against caller-supplied constraint predicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, get_args, get_origin

from .types import MediaObject, ParsedSignature, validate_type_annotation


# Output validation policy.

class ValidationMode(Enum):
    """
    How the runtime handles outputs that fail type or constraint validation.

    STRICT
        After the retry loop ends, invalid outputs are withheld. The runtime
        returns an ExecuteResult with result=None and failure diagnostics
        rather than raising for these output-validation failures.

    WARN
        Invalid outputs are returned with warnings and failure diagnostics.

    Passing validation means the output passed the applicable checks.
    It does not establish that the prediction satisfies the specification.

    Model-based self-checking is separate from these mechanical checks
    and does not establish semantic correctness.
    """

    STRICT = "strict"
    WARN = "warn"

    @classmethod
    def coerce(
        cls,
        value: ValidationMode | str | bool | None,
        default: ValidationMode | None = None,
    ) -> ValidationMode:
        """
        Resolve a mode, string value, or legacy boolean.

        True selects STRICT; False selects WARN.
        None uses the supplied default, falling back to WARN.
        """
        if value is None:
            return default if default is not None else cls.WARN
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.STRICT if value else cls.WARN
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


class UnsupportedMediaInputError(TypeError):
    """The prediction flow cannot supply a wrapper's content to the model."""


# Type-checking helpers.

def _is_media_type(t: Any) -> bool:
    try:
        return t is not Any and isinstance(t, type) and issubclass(t, MediaObject)
    except TypeError:
        return False


def check_value_type(value: Any, expected: Any, label: str = "value") -> None:
    """
    Check a value against a supported type annotation.

    Raises AITypeError on a detected mismatch.

    Supports primitive and container types, generic lists, dictionaries,
    tuples and sets, NoneType, Any, and MediaObject subclasses.
    """
    validate_type_annotation(expected)
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

    # Media types
    if _is_media_type(expected):
        if not isinstance(value, expected):
            raise AITypeError(
                f"{label}: expected {expected.__name__}, "
                f"got {type(value).__name__}"
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
                if len(value) != len(args):
                    raise AITypeError(
                        f"{label}: expected tuple of length {len(args)}, "
                        f"got length {len(value)}"
                    )
                for i, (item, t) in enumerate(zip(value, args)):
                    check_value_type(item, t, label=f"{label}[{i}]")
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

def validate_media_inputs(inputs: Dict[str, Any]) -> None:
    """Reject MediaObject wrappers, including Text, in nested input containers.

    Inspect dictionary keys and values and list/tuple/set/frozenset contents.
    Do not inspect arbitrary object attributes or infer media from strings or
    raw bytes. Track visited containers so cycles do not stall this check.
    """
    pending = [inputs]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if isinstance(value, MediaObject):
            raise UnsupportedMediaInputError(
                "MediaObject wrappers, including Text, are not supported by the "
                "text-only prediction flow; pass ordinary strings for text inputs."
            )

        if not isinstance(value, (dict, list, tuple, set, frozenset)):
            continue
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)

        if isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        else:
            pending.extend(value)


def validate_inputs(
    sig: ParsedSignature,
    inputs: Dict[str, Any],
    strict: bool = True,
) -> List[str]:
    """
    Check supplied argument names and values against a parsed signature.

    Returns warning strings in permissive mode.
    Strict mode raises AITypeError on the first detected mismatch.
    """
    warnings: List[str] = []
    declared_names = {p.name for p in sig.params}

    # Check for missing required arguments.
    for p in sig.params:
        if p.name not in inputs and p.default is None:
            msg = f"Missing required input: {p.name!r}"
            if strict:
                raise AITypeError(msg)
            warnings.append(msg)

    # Check for unexpected arguments.
    for key in inputs:
        if key not in declared_names:
            msg = f"Unexpected input: {key!r} (not in signature)"
            if strict:
                raise AITypeError(msg)
            warnings.append(msg)

    # Type-check each provided argument.
    for p in sig.params:
        if p.name not in inputs:
            continue
        try:
            check_value_type(
                inputs[p.name],
                p.resolved_type,
                label=f"input '{p.name}'",
            )
        except AITypeError as e:
            if strict:
                raise
            warnings.append(str(e))

    return warnings


# Output validation.

def validate_output(
    value: Any,
    return_type: Any,
    strict: bool = True,
) -> List[str]:
    """
    Check an output against an expected type annotation.

    Any or None disables this check; type(None) requires a None value.
    Strict mode raises AITypeError on a detected mismatch.
    Permissive mode returns the mismatch as a warning.
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
    """A named predicate on the output."""

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
    Check an output using caller-supplied predicates.

    Returns failure diagnostics in permissive mode.
    Strict mode raises AIConstraintError when a predicate returns a false value.
    Predicate exceptions propagate unchanged in either mode.
    """
    failures: List[str] = []
    for c in constraints:
        ok = c.check(value)
        if not ok:
            msg = c.message or f"Constraint {c.name!r} failed"
            if strict:
                raise AIConstraintError(msg)
            failures.append(msg)
    return failures