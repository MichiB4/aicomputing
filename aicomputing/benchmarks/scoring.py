"""Pure grading and answer normalization shared by benchmark runners."""

from __future__ import annotations

import math
import re
from typing import Any, List, Optional


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


def extract_gsm8k_answer(answer_str: str) -> float:
    """Read the numeric target following one explicit #### final-answer marker."""
    if not isinstance(answer_str, str) or answer_str.count("####") != 1:
        raise ValueError("GSM8K ground truth requires one #### final-answer marker")
    number = _finite_number(answer_str.split("####", 1)[1])
    if number is None:
        raise ValueError("GSM8K #### final answer must be wholly numeric and finite")
    return number


def answers_match(predicted: Any, expected: float, tolerance: float = 1e-3) -> bool:
    """Compare finite numeric answers using the existing absolute tolerance."""
    predicted_number = _finite_number(predicted)
    expected_number = _finite_number(expected)
    if predicted_number is None or expected_number is None:
        return False
    return abs(predicted_number - expected_number) < tolerance
