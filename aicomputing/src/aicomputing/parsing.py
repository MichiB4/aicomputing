"""
Parsing and type coercion for AI model outputs.

Handles: JSON extraction, Python literal parsing, bare scalars,
and type coercion to the declared return type.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
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


# Raised by the numeric/container coercions below when a value cannot be
# converted. OverflowError is the one that matters in practice: json.loads
# accepts the bare tokens Infinity / -Infinity / NaN, and a model emitting
# {"result": Infinity} against a `-> int` signature makes int(inf) raise
# OverflowError — which is neither a ValueError nor a TypeError, so nothing
# upstream caught it and it aborted the entire run.
_COERCION_ERRORS = (ValueError, TypeError, OverflowError)


def coerce_type(value: Any, target: Any, *, strict: bool = False) -> Any:
    """
    Best-effort coercion of *value* to *target* type.

    Total by contract: returns the coerced value, or the original value if
    coercion is not possible. This function must never raise — it is called
    from runtime.execute outside any try/except (runtime.py), so an exception
    here propagates out of VirtualRuntime.execute and aborts the caller's
    whole run rather than being recorded as one bad prediction.

    Coercion modes
    --------------
    strict=False (WARN / library)
        Repairs output into the declared type where it plausibly can:
        "5" -> 5, "yes" -> True, 3.7 -> 3.

    strict=True (the experiment)
        Only conversions that PRESERVE MEANING are applied - list -> tuple,
        int -> float, and float -> int when the float is exactly integral.
        Anything that would invent a value is left alone so that structural
        validation rejects it.

        This matters for what STRICT is supposed to assert. If the runtime
        repaired 3.7 into 3, or read "yes" as True, then a passing structural
        check would mean "the runtime managed to make this fit", not "the
        model returned something structurally compatible" - and the
        distinction is the whole point of the strict condition.
    """
    if target is None or target is Any:
        return value

    # Handle generic origins
    from typing import get_origin, get_args
    origin = get_origin(target)

    # Container coercions can fail on unhashable elements (a model returning
    # [[1],[2]] for a set[int]); fall back to the original value rather than
    # letting the TypeError escape.
    try:
        if origin is list and isinstance(value, (list, tuple)):
            args = get_args(target)
            if args:
                return [coerce_type(v, args[0], strict=strict) for v in value]
            return list(value)

        if origin is dict and isinstance(value, dict):
            args = get_args(target)
            if len(args) >= 2:
                return {coerce_type(k, args[0], strict=strict):
                        coerce_type(v, args[1], strict=strict)
                        for k, v in value.items()}
            return dict(value)

        # Tuple — JSON has no tuple type, so models return lists.
        # Convert list → tuple when the return type expects a tuple.
        if origin is tuple and isinstance(value, (list, tuple)):
            args = get_args(target)
            if args:
                if len(args) == 2 and args[1] is Ellipsis:
                    return tuple(coerce_type(v, args[0], strict=strict) for v in value)
                return tuple(coerce_type(v, t, strict=strict) for v, t in zip(value, args))
            return tuple(value)

        if origin is set and isinstance(value, (list, tuple, set)):
            args = get_args(target)
            if args:
                return {coerce_type(v, args[0], strict=strict) for v in value}
            return set(value)
    except _COERCION_ERRORS:
        return value

    # Plain types: int, float, str, bool, tuple
    if isinstance(target, type):
        # Plain tuple (not Tuple[int, int]) — convert list → tuple.
        # Representation-level: JSON has no tuple, so a model that returns the
        # right sequence has no way to express it. Meaning is preserved.
        if target is tuple and isinstance(value, (list, tuple)):
            return tuple(value)

        if isinstance(value, target):
            return value

        # Allow int where float is expected (standard Python widening; exact).
        if target is float and isinstance(value, int) and not isinstance(value, bool):
            return float(value)

        # float -> int only when the float is exactly integral. 3.0 -> 3 keeps
        # the value; 3.7 -> 3 invents one, so it is refused under STRICT and
        # left for validation to reject. See _strict_ok below.
        if target is int and isinstance(value, float):
            if value.is_integer():
                try:
                    return int(value)
                except _COERCION_ERRORS:
                    pass
            elif strict:
                return value            # let validation call it a type failure

        if not strict:
            # ---- WARN-only repairs -------------------------------------
            # These change the value's meaning, or read intent out of text.
            # Useful for interactive and library use, where a nearly-right
            # answer beats an exception. Never used in the experiment: under
            # STRICT they would let the runtime manufacture compatibility
            # that the model did not actually produce.
            if target is float and isinstance(value, str):
                try:
                    return float(value)
                except _COERCION_ERRORS:
                    pass
            if target is int and isinstance(value, (float, str)):
                try:
                    return int(value)
                except _COERCION_ERRORS:
                    pass
            if target is bool:
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes")
                try:
                    return bool(value)
                except _COERCION_ERRORS:
                    return value

        if target is str and not strict:
            # WARN only. str(value) is lossless as a mapping but it is not
            # meaning-preserving as a TYPE claim: turning True into "True"
            # because the signature says -> str reports a boolean answer as a
            # string one, and STRICT would then pass a value the model never
            # produced in that form.
            try:
                return str(value)
            except _COERCION_ERRORS:
                return value

        if target is bool and isinstance(value, bool):
            return value

    return value


def parse_result(text: str, return_type: Optional[Any] = None,
                 *, strict: bool = False) -> Any:
    """
    Parse model output text into a Python value.

    1. Try JSON extraction ({"result": ...})
    2. Try ast.literal_eval on the whole text (or last line)
    3. Fall back to scalar parsing
    4. Optionally coerce to return_type

    Delegates to parse_result_detailed() and discards the status, so the two
    cannot drift apart. Use parse_result_detailed() when you need to know
    whether the value came from a structured form or from the raw-text
    fallback.
    """
    return parse_result_detailed(text, return_type, strict=strict).value


# ---------------------------------------------------------------------------
# Parse reporting
# ---------------------------------------------------------------------------

@dataclass
class ParseOutcome:
    """
    A parsed value plus how it was obtained.

    parse_result() is deliberately permissive: when nothing structured can be
    recovered it returns the cleaned text as a string. That is the right
    behaviour for benchmarks whose answers really are strings (BBH, HotpotQA),
    but it means a malformed response and a legitimate one-word answer are
    indistinguishable downstream -- garbage is scored as an ordinary wrong
    answer, and the model looks capable-but-wrong rather than unparseable.

    The status says which path produced the value:

      "parsed"           recovered from a structured form the prompt asked
                         for: {"result": ...}, or a bare Python/JSON literal.
      "fallback_parsed"  no structured form was found; the cleaned text is
                         being returned as a string. Legitimate for a
                         string-valued specification, suspicious otherwise.
      "failed"           nothing at all could be recovered (empty response).

    This is metadata, not a verdict. A "fallback_parsed" string answer to a
    `-> str` signature is perfectly normal, so nothing here decides
    correctness on its own.
    """

    value: Any
    status: str                      # "parsed" | "fallback_parsed" | "failed"
    error: Optional[str] = None

    PARSED = "parsed"
    FALLBACK = "fallback_parsed"
    FAILED = "failed"


def parse_result_detailed(text: str, return_type: Optional[Any] = None,
                          *, strict: bool = False) -> ParseOutcome:
    """
    parse_result(), but reporting how the value was obtained.

    Structured recovery is attempted exactly as parse_result does; the two
    stay in step because this function performs the recovery and
    parse_result() delegates to it.
    """
    if not text or not text.strip():
        return ParseOutcome(None, ParseOutcome.FAILED, "empty response")

    coerce = lambda v: (coerce_type(v, return_type, strict=strict)
                        if return_type is not None else v)

    # 1. {"result": ...} or any single-key JSON object.
    val = extract_json_result(text)
    if val is not _MISSING:
        return ParseOutcome(coerce(val), ParseOutcome.PARSED)

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:python|json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # 2. A bare literal, whole-text then last meaningful line.
    try:
        return ParseOutcome(coerce(ast.literal_eval(cleaned)), ParseOutcome.PARSED)
    except (ValueError, SyntaxError):
        pass

    for line in reversed(cleaned.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return ParseOutcome(coerce(ast.literal_eval(line)), ParseOutcome.PARSED)
        except (ValueError, SyntaxError):
            pass
        line_val = extract_json_result(line)
        if line_val is not _MISSING:
            return ParseOutcome(coerce(line_val), ParseOutcome.PARSED)
        break

    # 3. Scalar fallback. parse_python_literal still recognises barewords
    #    (none/true/false) and numerics, which count as parsed; anything else
    #    comes back as the raw string and is reported as a fallback.
    val = parse_python_literal(cleaned)
    status = (ParseOutcome.FALLBACK if isinstance(val, str) and val == cleaned
              else ParseOutcome.PARSED)
    error = "no structured value found; returned raw text" if status == ParseOutcome.FALLBACK else None
    return ParseOutcome(coerce(val), status, error)
