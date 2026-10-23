"""Recover model predictions from response envelopes, literals, or plain text.

A dictionary whose sole key is "result" is treated as an envelope. Other
objects are preserved as values. Embedded containers are decoded whole;
nested fragments of malformed containers are never selected as answers.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from typing import Any, Iterator, Optional, get_args, get_origin


class _MissingSentinel:
    """Distinguish missing recovery from a successfully recovered None."""

    def __repr__(self) -> str:
        return "<MISSING>"

    def __bool__(self) -> bool:
        return False


_MISSING = _MissingSentinel()
_COERCION_ERRORS = (ValueError, TypeError, OverflowError)


def _strip_code_fences(text: str) -> str:
    """Strip outer whitespace and one complete bare/python/json fence block."""
    cleaned = text.strip()
    lines = cleaned.splitlines(keepends=True)
    if not lines or lines[0].strip() not in ("```", "```python", "```json"):
        return cleaned
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "```":
            if index == len(lines) - 1:
                return "".join(lines[1:index]).strip()
            break
    return cleaned


def _decode_literal(text: str) -> Any:
    """Decode one complete JSON value or Python literal without executing code."""
    try:
        return json.loads(text)
    except ValueError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError):
        return _MISSING


def _unwrap_result(value: Any) -> Any:
    """Unwrap a result-only envelope; preserve other dictionaries."""
    if isinstance(value, dict) and len(value) == 1 and "result" in value:
        return value["result"]
    return value


def _container_spans(text: str) -> Iterator[str]:
    """Locate outer object/array spans, respecting nested brackets and strings.

    This identifies boundaries only; JSON/Python decoders validate syntax.
    An incomplete final span is yielded whole so it cannot be replaced by
    a parseable inner fragment or an earlier example.
    """
    start: Optional[int] = None
    stack: list[str] = []
    quote: Optional[str] = None
    escaped = False
    skip_until = 0
    closing = {"}": "{", "]": "[", ")": "("}

    for index, char in enumerate(text):
        if index < skip_until:
            continue
        if start is None:
            if char in "{[":
                start = index
                stack = [char]
            continue

        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif text.startswith(quote, index):
                skip_until = index + len(quote)
                quote = None
            continue

        if char in ("'", '\"'):
            quote = char * 3 if text.startswith(char * 3, index) else char
            skip_until = index + len(quote)
        elif char in "{[(":
            stack.append(char)
        elif char in "}])":
            if not stack or stack[-1] != closing[char]:
                # A malformed outer container cannot supply an inner answer.
                yield text[start:]
                return
            stack.pop()
            if not stack:
                yield text[start:index + 1]
                start = None

    if start is not None:
        yield text[start:]


def _recover_structured(
    text: str,
    *,
    unwrap_result: bool = True,
) -> tuple[Any, Optional[str]]:
    """Decode a whole response, or its last outer container in prose."""
    value = _decode_literal(text)
    if value is not _MISSING:
        return (_unwrap_result(value) if unwrap_result else value), None

    found = False
    value = _MISSING
    error = None
    for candidate in _container_spans(text):
        found = True
        decoded = _decode_literal(candidate)
        if decoded is _MISSING:
            value = _MISSING
            error = "malformed or incomplete structured answer"
        else:
            value = _unwrap_result(decoded) if unwrap_result else decoded
            error = None

    if found:
        return value, error
    return _MISSING, None


def extract_json_result(text: str) -> Any:
    """Recover a complete value or result envelope; return _MISSING on failure.

    Supports JSON and Python literals. Bare dictionaries are preserved.
    When prose contains multiple outer containers, the last one is used.
    """
    value, _ = _recover_structured(_strip_code_fences(text))
    return value


def parse_python_literal(text: str) -> Any:
    """Recover a literal or scalar, otherwise return cleaned text."""
    cleaned = _strip_code_fences(text)
    value = _decode_literal(cleaned)
    if value is not _MISSING:
        return value

    lower = cleaned.lower()
    if lower in ("none", "null"):
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if "." in cleaned or "e" in lower:
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return cleaned


def coerce_type(value: Any, target: Any, *, strict: bool = False) -> Any:
    """Apply the agreed limited conversions; validate the result separately.

    Conversion rules are identical under WARN and STRICT. The strict keyword
    remains accepted for compatibility with existing callers.
    Unsupported or unsuccessful conversions retain the original value.
    Dictionary keys and sets are never coerced, avoiding entry loss.
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

    Values may already be coerced. Status does not establish type validity
    or semantic correctness. None can be a successfully parsed value.
    """

    value: Any
    status: str
    error: Optional[str] = None

    PARSED = "parsed"
    FALLBACK = "fallback_parsed"
    FAILED = "failed"


def parse_result_detailed(
    text: str,
    return_type: Optional[Any] = None,
    *,
    strict: bool = False,
    unwrap_result: bool = True,
) -> ParseOutcome:
    """Recover a prediction and optionally coerce it to the expected type.

    Whole-response literals take precedence over embedded containers.
    Embedded containers are considered in order, with the last one selected.
    Malformed selected containers fail rather than exposing inner fragments.
    Without containers, the last non-empty line may supply a literal answer;
    otherwise scalar recovery or whole-text fallback is used.

    The strict keyword remains accepted for compatibility; conversion rules
    are identical under WARN and STRICT.

    Set unwrap_result=False to preserve a complete response object, such as
    a contract verdict containing both "verdict" and "result" fields.
    """
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return ParseOutcome(None, ParseOutcome.FAILED, "empty response")

    value, error = _recover_structured(cleaned, unwrap_result=unwrap_result)
    if error is not None:
        return ParseOutcome(None, ParseOutcome.FAILED, error)
    if value is not _MISSING:
        return ParseOutcome(
            coerce_type(value, return_type, strict=strict),
            ParseOutcome.PARSED,
        )

    # With no structured container, try the final non-empty line as a literal.
    last_line = next(line.strip() for line in reversed(cleaned.splitlines()) if line.strip())
    value = _decode_literal(last_line)
    if value is not _MISSING:
        return ParseOutcome(
            coerce_type(value, return_type, strict=strict),
            ParseOutcome.PARSED,
        )

    value = parse_python_literal(cleaned)
    fallback = isinstance(value, str) and value == cleaned
    return ParseOutcome(
        coerce_type(value, return_type, strict=strict),
        ParseOutcome.FALLBACK if fallback else ParseOutcome.PARSED,
        "no structured value found; returned cleaned text" if fallback else None,
    )


def parse_result(
    text: str,
    return_type: Optional[Any] = None,
    *,
    strict: bool = False,
) -> Any:
    """Return the recovered value, discarding parsing status and diagnostics.

    Use parse_result_detailed() to distinguish valid None from failure.
    """
    return parse_result_detailed(text, return_type, strict=strict).value
