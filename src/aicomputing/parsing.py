"""
Parsing and type coercion for AI model outputs.

Handles: JSON extraction, Python literal parsing, bare scalars,
and type coercion to the declared return type.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Optional


def extract_json_result(text: str) -> Any:
    """
    Extract the predicted value from JSON embedded in *text*.

    Priority:
    1. Look for {"result": ...} — our standard format
    2. Fall back to any single-key JSON object (e.g. {"groups": [...]})
    3. Fall back to any JSON object and return its first value

    Returns a sentinel _MISSING (not None) when extraction fails,
    so we can distinguish "result is None" from "no result found".
    """
    # Collect all JSON objects in the text
    all_candidates = []

    # Strategy 1: flat JSON objects
    for m in re.finditer(r"\{[^{}]*\}", text):
        all_candidates.append(m.group(0))

    # Strategy 2: nested JSON objects
    for m in re.finditer(r"\{[\s\S]*\}", text):
        chunk = m.group(0)
        if chunk not in all_candidates:
            all_candidates.append(chunk)

    # First pass: look for "result" key specifically
    result_candidates = [c for c in all_candidates
                         if '"result"' in c or "'result'" in c]
    result_candidates.sort(key=len)
    for cand in result_candidates:
        val = _try_parse_json_obj(cand, key="result")
        if val is not _MISSING:
            return val

    # Second pass: any parseable JSON object — take the first value
    all_candidates.sort(key=len)
    for cand in all_candidates:
        val = _try_parse_any_json_value(cand)
        if val is not _MISSING:
            return val

    return _MISSING


class _MissingSentinel:
    """Sentinel to distinguish 'no result found' from 'result is None'."""
    def __repr__(self):
        return "<MISSING>"
    def __bool__(self):
        return False

_MISSING = _MissingSentinel()


def _try_parse_json_obj(text: str, key: str = "result") -> Any:
    """Try to parse a JSON object and extract a specific key."""
    for attempt in [text, text.replace("'", '"')]:
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict) and key in obj:
                return obj[key]
        except (json.JSONDecodeError, ValueError):
            pass

    # Try ast.literal_eval (handles Python dict syntax: {'result': [1,2,3]})
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict) and key in obj:
            return obj[key]
    except (ValueError, SyntaxError):
        pass

    return _MISSING


def _try_parse_any_json_value(text: str) -> Any:
    """
    Parse a JSON object and return its first (or only) value.
    Handles cases where the model uses a key other than "result",
    e.g. {"groups": [...]} or {"answer": 42}.
    """
    for attempt in [text, text.replace("'", '"')]:
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict) and len(obj) == 1:
                return next(iter(obj.values()))
            # If multiple keys, still try — return the first value
            if isinstance(obj, dict) and obj:
                return next(iter(obj.values()))
        except (json.JSONDecodeError, ValueError):
            pass

    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict) and obj:
            return next(iter(obj.values()))
    except (ValueError, SyntaxError):
        pass

    return _MISSING


def parse_python_literal(text: str) -> Any:
    """
    Safely parse a Python literal from text.
    Handles: ints, floats, strings, bools, None, lists, dicts, tuples.
    """
    s = text.strip()

    # Strip markdown code fences
    s = re.sub(r"^```(?:python|json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()

    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass

    # Common barewords
    lower = s.lower()
    if lower == "none" or lower == "null":
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False

    # Numeric
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        pass

    # Return as string if nothing else works
    return s


def coerce_type(value: Any, target: Any) -> Any:
    """
    Best-effort coercion of *value* to *target* type.
    Returns the coerced value, or the original if coercion fails.
    """
    if target is None or target is Any:
        return value

    # Handle generic origins
    from typing import get_origin, get_args
    origin = get_origin(target)

    if origin is list and isinstance(value, (list, tuple)):
        args = get_args(target)
        if args:
            return [coerce_type(v, args[0]) for v in value]
        return list(value)

    if origin is dict and isinstance(value, dict):
        args = get_args(target)
        if len(args) >= 2:
            return {coerce_type(k, args[0]): coerce_type(v, args[1]) for k, v in value.items()}
        return dict(value)

    # Tuple — JSON has no tuple type, so models return lists.
    # Convert list → tuple when the return type expects a tuple.
    if origin is tuple and isinstance(value, (list, tuple)):
        args = get_args(target)
        if args:
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(coerce_type(v, args[0]) for v in value)
            return tuple(coerce_type(v, t) for v, t in zip(value, args))
        return tuple(value)

    if origin is set and isinstance(value, (list, tuple, set)):
        args = get_args(target)
        if args:
            return {coerce_type(v, args[0]) for v in value}
        return set(value)

    # Plain types: int, float, str, bool, tuple
    if isinstance(target, type):
        # Plain tuple (not Tuple[int, int]) — convert list → tuple
        if target is tuple and isinstance(value, (list, tuple)):
            return tuple(value)

        if isinstance(value, target):
            return value

        # Allow int where float is expected (standard Python widening)
        if target is float and isinstance(value, (int, str)):
            try:
                return float(value)
            except (ValueError, TypeError):
                pass
        if target is int and isinstance(value, (float, str)):
            try:
                return int(value)
            except (ValueError, TypeError):
                pass
        if target is str:
            return str(value)

        # Bool — handle case-insensitive strings and JSON true/false
        if target is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)

    return value


def parse_result(text: str, return_type: Optional[Any] = None) -> Any:
    """
    Parse model output text into a Python value.

    1. Try JSON extraction ({"result": ...})
    2. Try ast.literal_eval on the whole text (or last line)
    3. Fall back to scalar parsing
    4. Optionally coerce to return_type
    """
    # Try JSON first
    val = extract_json_result(text)
    if val is not _MISSING:
        if return_type is not None:
            val = coerce_type(val, return_type)
        return val

    # Strip markdown fences from the whole text
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:python|json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Try ast.literal_eval on the whole cleaned text
    try:
        val = ast.literal_eval(cleaned)
        if return_type is not None:
            val = coerce_type(val, return_type)
        return val
    except (ValueError, SyntaxError):
        pass

    # Try the last non-empty line (model may have prose before the answer)
    lines = cleaned.splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            val = ast.literal_eval(line)
            if return_type is not None:
                val = coerce_type(val, return_type)
            return val
        except (ValueError, SyntaxError):
            pass
        # Try JSON on this line too
        line_val = extract_json_result(line)
        if line_val is not _MISSING:
            if return_type is not None:
                line_val = coerce_type(line_val, return_type)
            return line_val
        break  # only try the last meaningful line

    # Fall back to simple scalar parsing
    val = parse_python_literal(cleaned)
    if return_type is not None:
        val = coerce_type(val, return_type)
    return val
