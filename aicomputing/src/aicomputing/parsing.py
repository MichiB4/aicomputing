"""Decode one framed prediction, preserving isolated malformed result text.

Parsing recovers values and diagnostics. Type conversion and output validation
are handled separately by strategies and the runtime.
"""
from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Optional, get_args, get_origin

from .answer_protocol import FINAL_START, FINAL_END
from .status import ParseStatus

_MISSING = object()
_COERCION_ERRORS = (ValueError, TypeError, OverflowError)


def coerce_type(value: Any, target: Any) -> Any:
    """Apply limited scalar/container conversions; validate the result separately.

    Lists and tuples convert between those container types and recursively
    convert annotated elements. Fixed-length tuple targets require matching
    lengths. Dictionaries recursively convert annotated values, leaving keys
    unchanged. Sets are never converted.

    Strings convert to bool only for stripped, case-insensitive true/false;
    to int using int(); and to float only when float() produces a finite value.
    Integral floats convert to int. Integers convert to float only when the
    converted value equals the original. Booleans never convert to numbers.

    Conversion rules are identical under WARN and STRICT.
    Unsupported or unsuccessful conversions retain the original value.
    """
    if target is None or target is Any:
        return value

    container = get_origin(target) or target
    args = get_args(target)
    try:
        if container is list and isinstance(value, (list, tuple)):
            if args:
                return [coerce_type(item, args[0]) for item in value]
            return list(value)

        if container is tuple and isinstance(value, (list, tuple)):
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(coerce_type(item, args[0]) for item in value)
            if args:
                if len(value) != len(args):
                    return value
                return tuple(
                    coerce_type(item, item_type)
                    for item, item_type in zip(value, args)
                )
            return tuple(value)

        if container is dict and isinstance(value, dict):
            if len(args) >= 2:
                return {
                    key: coerce_type(item, args[1])
                    for key, item in value.items()
                }
            return value

        if target is bool and isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "true":
                return True
            if normalized == "false":
                return False
            return value

        # bool inherits int, but is not an allowed numeric conversion.
        if isinstance(value, bool):
            return value

        if target is int:
            if isinstance(value, str):
                return int(value)
            if isinstance(value, float) and value.is_integer():
                return int(value)

        if target is float:
            if isinstance(value, int):
                converted = float(value)
                return converted if converted == value else value
            if isinstance(value, str):
                converted = float(value)
                return converted if math.isfinite(converted) else value
    except _COERCION_ERRORS:
        return value

    return value

@dataclass
class ParseOutcome:
    """Recovered value, parsing path, and optional diagnostic.

    Values retain their decoded types. Status does not establish type validity
    or semantic correctness. None can be a successfully parsed value.
    """

    value: Any
    status: ParseStatus
    error: Optional[str] = None

    def __post_init__(self) -> None:
        self.status = ParseStatus(self.status)


class _DuplicateField(ValueError):
    pass

def _decode_framed_literal(payload):
    """Decode data once, refusing duplicate keys and nonliteral expressions."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateField("duplicate object key")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("non-finite JSON number")

    try:
        return json.loads(payload, object_pairs_hook=unique, parse_constant=reject_constant)
    except _DuplicateField:
        raise
    except ValueError:
        pass
    try:
        tree = ast.parse(payload, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys = set()
                for key in node.keys:
                    value = ast.literal_eval(key)
                    if value in keys:
                        raise _DuplicateField("duplicate object key")
                    keys.add(value)
        return ast.literal_eval(tree)
    except _DuplicateField:
        raise
    except (ValueError, SyntaxError, TypeError):
        return _MISSING

def _recover_result_text(payload, *, unwrap_result):
    """Preserve one explicit result field; never select a nested fragment."""
    field = r"""(?:"result"|'result')\s*:\s*(?P<answer>.*?)"""
    verdict = r"""(?:"verdict"|'verdict')\s*:\s*(?:"(?P<j>accept|revise)"|'(?P<p>accept|revise)')"""
    separator = r"\s*,\s*"
    fields = field if unwrap_result else verdict + separator + field
    match = re.fullmatch(
        r"\{\s*" + fields + r"\s*\}",
        payload, flags=re.DOTALL,
    )
    verdict_after_result = False
    if match is None and not unwrap_result:
        match = re.fullmatch(
            r"\{\s*" + field + separator + verdict + r"\s*\}",
            payload, flags=re.DOTALL,
        )
        verdict_after_result = True
    if match is None:
        return None
    answer = match["answer"].strip()
    if not answer:
        return None
    # Reject extra outer fields, mismatched delimiters and unterminated
    # strings. Unclosed list/tuple delimiters remain in the returned text.
    stack, quote, escaped, skip_until = [], None, False, 0
    closing = {"}": "{", "]": "[", ")": "("}
    for index, char in enumerate(answer):
        if index < skip_until:
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif answer.startswith(quote, index):
                skip_until, quote = index + len(quote), None
        elif char in ("'", '"'):
            quote = char * 3 if answer.startswith(char * 3, index) else char
            skip_until = index + len(quote)
        elif char in "{[(":
            stack.append(char)
        elif char in "}])":
            if not stack or stack.pop() != closing[char]:
                return None
        elif char == "," and not stack:
            return None
    if quote is not None or "{" in stack:
        return None
    # A trailing verdict must lie outside the result's containers. Otherwise
    # its separator could belong to an unfinished list or tuple.
    if verdict_after_result and stack:
        return None
    value = answer if unwrap_result else {
        "verdict": match["j"] or match["p"], "result": answer,
    }
    return ParseOutcome(value, ParseStatus.FALLBACK,
                        "malformed result literal; preserved result-field text without repair")

def parse_result(text: str, *, unwrap_result: bool = True) -> ParseOutcome:
    """Decode one complete final envelope, independently of earlier prose.

    Markers are reserved standalone lines. There is deliberately no recovery
    from missing boundaries, inner fragments, duplicate keys or extra text.
    One isolated malformed result field is retained as text, with a diagnostic.
    No punctuation, digits or delimiters are repaired.
    """
    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip() == FINAL_START]
    ends = [index for index, line in enumerate(lines) if line.strip() == FINAL_END]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        return ParseOutcome(None, ParseStatus.FAILED, "require exactly one complete final-answer block")
    if any(line.strip() for line in lines[ends[0] + 1:]):
        return ParseOutcome(None, ParseStatus.FAILED, "text follows final-answer block")
    payload = "\n".join(lines[starts[0] + 1:ends[0]]).strip()
    try:
        value = _decode_framed_literal(payload)
    except _DuplicateField:
        return ParseOutcome(None, ParseStatus.FAILED, "duplicate object key")
    def supported(item):
        if item is None or type(item) in (bool, int, str, bytes):
            return True
        if type(item) is float:
            return math.isfinite(item)
        if type(item) is dict:
            return all(supported(key) and supported(val) for key, val in item.items())
        if type(item) in (list, tuple, set):
            return all(supported(part) for part in item)
        return False

    if value is _MISSING or not supported(value):
        recovered = _recover_result_text(payload, unwrap_result=unwrap_result)
        if recovered is not None:
            return recovered
        return ParseOutcome(None, ParseStatus.FAILED, "final-answer block must contain one complete literal object")
    if type(value) is not dict:
        return ParseOutcome(None, ParseStatus.FAILED, "final-answer envelope must be an object")
    required = {"result"} if unwrap_result else {"verdict", "result"}
    if set(value) != required:
        return ParseOutcome(None, ParseStatus.FAILED, "missing or unexpected final-answer fields")
    if not unwrap_result:
        verdict = value["verdict"]
        if type(verdict) is not str or verdict not in ("accept", "revise"):
            return ParseOutcome(None, ParseStatus.FAILED, "invalid framed contract verdict")
        return ParseOutcome(value, ParseStatus.PARSED)
    result = value["result"]
    return ParseOutcome(result, ParseStatus.PARSED)
