"""Pure grading under the reviewed release's frozen scoring profiles."""

from __future__ import annotations

import math
import re
from typing import Any, List, Optional

from .values import canonical_json

PROFILES = {"structured_abs_1e-6", "numeric_abs_1e-3", "unordered_set_abs_1e-6",
            "mbpp_exact_or_float_v1", "mbpp_float_atol_1e-4_v1"}


def compare(predicted: Any, expected: Any, profile: str) -> bool:
    """Grade one prediction with its task's scoring profile."""
    if profile == "numeric_abs_1e-3":
        return answers_match(predicted, expected)
    if profile in ("mbpp_exact_or_float_v1", "mbpp_float_atol_1e-4_v1"):
        return _mbpp_float_compare(predicted, expected,
                                   1e-4 if profile == "mbpp_float_atol_1e-4_v1" else 1e-6)
    if profile == "structured_abs_1e-6":
        return results_match(predicted, expected)
    if profile == "unordered_set_abs_1e-6":
        # Only tasks whose published scorer specifies set equivalence may
        # request this. Order and duplicate count are intentionally ignored.
        if type(predicted) not in (list, tuple, set, frozenset) or type(expected) not in (list, tuple, set, frozenset):
            return False
        try:
            return results_match(set(predicted), set(expected))
        except TypeError:
            return False
    raise ValueError(f"Unknown scoring profile: {profile}")


def _unordered_results_match(predicted: List[Any], expected: List[Any], tolerance: float) -> bool:
    """Find a one-to-one match without depending on container iteration order."""
    if len(predicted) != len(expected):
        return False
    candidates = [
        [index for index, value in enumerate(expected)
         if results_match(item, value, tolerance)]
        for item in predicted
    ]
    matches: List[Optional[int]] = [None] * len(expected)

    def assign(item_index: int, visited: set) -> bool:
        for index in candidates[item_index]:
            if index in visited:
                continue
            visited.add(index)
            previous = matches[index]
            if previous is None or assign(previous, visited):
                matches[index] = item_index
                return True
        return False

    return all(assign(index, set()) for index in range(len(predicted)))


def results_match(predicted: Any, expected: Any, tolerance: float = 1e-6) -> bool:
    """Compare values recursively, without stringification or boolean coercion.

    Integers and floats may match numerically using the absolute tolerance;
    list/tuple sequences may match across those two container types. Dictionary
    entries and set/frozenset members match one-to-one without regard to order.
    Strings retain case and whitespace. Nonfinite floats never match.
    Other value types require matching types and their normal equality.
    """
    if predicted is None or expected is None:
        return predicted is expected
    if isinstance(predicted, bool) or isinstance(expected, bool):
        return isinstance(predicted, bool) and isinstance(expected, bool) and predicted == expected
    if isinstance(predicted, (int, float)) or isinstance(expected, (int, float)):
        if not isinstance(predicted, (int, float)) or not isinstance(expected, (int, float)):
            return False
        if any(isinstance(value, float) and not math.isfinite(value)
               for value in (predicted, expected)):
            return False
        if predicted == expected:
            return True
        try:
            return abs(predicted - expected) < tolerance
        except OverflowError:
            return False
    if isinstance(predicted, (list, tuple)) or isinstance(expected, (list, tuple)):
        return (
            isinstance(predicted, (list, tuple)) and isinstance(expected, (list, tuple))
            and len(predicted) == len(expected)
            and all(results_match(left, right, tolerance) for left, right in zip(predicted, expected))
        )
    if isinstance(predicted, dict) or isinstance(expected, dict):
        return (
            isinstance(predicted, dict) and isinstance(expected, dict)
            and _unordered_results_match(list(predicted.items()), list(expected.items()), tolerance)
        )
    if isinstance(predicted, (set, frozenset)) or isinstance(expected, (set, frozenset)):
        return (
            isinstance(predicted, (set, frozenset)) and isinstance(expected, (set, frozenset))
            and _unordered_results_match(list(predicted), list(expected), tolerance)
        )
    return type(predicted) is type(expected) and bool(predicted == expected)


_NUMBER_PATTERN = re.compile(
    r"[+-]?(?:(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]*)?|\.[0-9]+)"
    r"(?:[eE][+-]?[0-9]+)?"
)


def _finite_number(value: Any) -> Optional[float]:
    """Read a finite number or wholly numeric text, never a boolean or prose."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        if _NUMBER_PATTERN.fullmatch(text) is None:
            return None
        value = text.replace(",", "")
    elif not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def answers_match(predicted: Any, expected: float, tolerance: float = 1e-3) -> bool:
    """Compare finite numeric answers using the existing absolute tolerance."""
    predicted_number = _finite_number(predicted)
    expected_number = _finite_number(expected)
    if predicted_number is None or expected_number is None:
        return False
    return abs(predicted_number - expected_number) < tolerance


def _mbpp_float_compare(predicted, expected, atol):
    """Published MBPP float tolerances, with strict integer and boolean identity.

    Approximation applies to float targets; integer labels remain exact.
    Sequence ordering and duplicate counts are preserved. Set-equivalence tasks
    use their separate profile. This does not execute the upstream scorer.
    """
    if type(expected) is float:
        if type(predicted) not in (int, float):
            return False
        try:
            return math.isfinite(predicted) and math.isfinite(expected) and math.isclose(
                predicted, expected, rel_tol=1e-7, abs_tol=atol)
        except OverflowError:
            return False
    if type(expected) in (list, tuple):
        return (type(predicted) in (list, tuple) and len(predicted) == len(expected)
                and all(_mbpp_float_compare(left, right, atol)
                        for left, right in zip(predicted, expected)))
    if type(expected) is dict:
        if type(predicted) is not dict or len(predicted) != len(expected):
            return False
        # Keys retain their original typed identity; approximation belongs to
        # numeric values, not dictionary key selection.
        try:
            keys = {canonical_json(key): key for key in predicted}
        except (TypeError, ValueError):
            return False
        return all((encoded := canonical_json(key)) in keys
                   and _mbpp_float_compare(predicted[keys[encoded]], value, atol)
                   for key, value in expected.items())
    if type(expected) is int:
        return type(predicted) in (int, float) and predicted == expected
    return results_match(predicted, expected)
