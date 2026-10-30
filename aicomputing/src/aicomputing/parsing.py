"""Recover model predictions from response envelopes, literals, or plain text.

A dictionary whose sole key is "result" is treated as an envelope. Other
objects are preserved as values. Embedded containers are decoded whole;
nested fragments of malformed containers are never selected as answers.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterator, Optional, get_args, get_origin

from .answer_protocol import FRAMED_V1, FRAMED_TEXT_V1, LEGACY, FINAL_START, FINAL_END, validate_answer_format


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


def _container_spans(text: str) -> Iterator[Optional[str]]:
    """Locate outer data spans, respecting nested brackets and strings.

    This identifies boundaries only; JSON/Python decoders validate syntax.
    An incomplete final span is yielded whole so it cannot be replaced by
    a parseable inner fragment or an earlier example.

    Outside data, a line labelled State: is reasoning rather than an answer.
    Yield None for it so a trace without a later answer cannot select an
    earlier example. A single bracket quoted in inline Markdown is prose.
    Neither rule applies inside a data container or literal string.
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
            if char == "S" and text.startswith("State:", index):
                line_start = max(text.rfind("\n", 0, index), text.rfind("\r", 0, index)) + 1
                if not text[line_start:index].strip():
                    line_end = index + len("State:")
                    while line_end < len(text) and text[line_end] not in "\r\n":
                        line_end += 1
                    skip_until = line_end
                    yield None
                    continue
            if (char == "`" and not text.startswith("``", index)
                    and (index == 0 or text[index - 1] != "`")):
                end = text.find("`", index + 1)
                if end != -1 and (end + 1 == len(text) or text[end + 1] != "`"):
                    code = text[index + 1:end]
                    if ("\r" not in code and "\n" not in code
                            and len(code.strip()) == 1 and code.strip() in "{}[]()"):
                        skip_until = end + 1
                        continue
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
        if candidate is None:
            value = _MISSING
            error = "reasoning trace has no final structured answer"
            continue
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
        return _MISSING
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
                        return _MISSING
                    keys.add(value)
        return ast.literal_eval(tree)
    except (ValueError, SyntaxError, TypeError):
        return _MISSING


def _framed_text_fallback(payload, *, unwrap_result):
    """Preserve one explicit result field; never select a nested fragment."""
    field = r"""(?:"result"|'result')\s*:\s*(?P<answer>.*?)"""
    verdict = r"""(?:"verdict"|'verdict')\s*:\s*(?:"(?P<j>accept|revise)"|'(?P<p>accept|revise)')\s*,\s*"""
    match = re.fullmatch(
        r"\{\s*" + ("" if unwrap_result else verdict) + field + r"\s*\}",
        payload, flags=re.DOTALL,
    )
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
    value = answer if unwrap_result else {
        "verdict": match["j"] or match["p"], "result": answer,
    }
    return ParseOutcome(value, ParseOutcome.FALLBACK,
                        "malformed result literal; preserved result-field text without repair")


def _parse_framed_result(text, return_type, *, unwrap_result, strict, preserve_text=False):
    """Decode one complete final envelope, independently of earlier prose.

    Markers are reserved standalone lines. There is deliberately no recovery
    from missing boundaries, inner fragments, duplicate keys or extra text.
    Legacy recovery is never attempted. The opt-in text protocol may retain
    one isolated malformed result field as text, with a diagnostic.
    """
    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip() == FINAL_START]
    ends = [index for index, line in enumerate(lines) if line.strip() == FINAL_END]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        return ParseOutcome(None, ParseOutcome.FAILED, "require exactly one complete final-answer block")
    if any(line.strip() for line in lines[ends[0] + 1:]):
        return ParseOutcome(None, ParseOutcome.FAILED, "text follows final-answer block")
    payload = "\n".join(lines[starts[0] + 1:ends[0]]).strip()
    value = _decode_framed_literal(payload)
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
        if preserve_text:
            fallback = _framed_text_fallback(payload, unwrap_result=unwrap_result)
            if fallback is not None:
                return fallback
        return ParseOutcome(None, ParseOutcome.FAILED, "final-answer block must contain one complete literal object")
    if type(value) is not dict:
        return ParseOutcome(None, ParseOutcome.FAILED, "final-answer envelope must be an object")
    required = {"result"} if unwrap_result else {"verdict", "result"}
    if set(value) != required:
        return ParseOutcome(None, ParseOutcome.FAILED, "missing or unexpected final-answer fields")
    if not unwrap_result:
        verdict = value["verdict"]
        if type(verdict) is not str or verdict not in ("accept", "revise"):
            return ParseOutcome(None, ParseOutcome.FAILED, "invalid framed contract verdict")
        return ParseOutcome(value, ParseOutcome.PARSED)
    result = value["result"] if preserve_text else coerce_type(value["result"], return_type, strict=strict)
    return ParseOutcome(result, ParseOutcome.PARSED)


def parse_result_detailed(
    text: str,
    return_type: Optional[Any] = None,
    *,
    strict: bool = False,
    unwrap_result: bool = True,
    answer_format: str = LEGACY,
) -> ParseOutcome:
    """Recover a prediction and optionally coerce it to the expected type.

    framed_v1 requires one complete, final, explicitly delimited envelope;
    preceding explanation text is not scanned for answer candidates.
    The following recovery rules apply only to the default legacy format.

    Whole-response literals take precedence over embedded containers.
    Embedded containers are considered in order, with the last one selected.
    Malformed selected containers fail rather than exposing inner fragments.
    State: trace lines outside data require a later structured answer; a
    bare scalar after such a trace is not an explicit final result object.
    Without containers, the last non-empty line may supply a literal answer;
    otherwise scalar recovery or whole-text fallback is used.

    The strict keyword remains accepted for compatibility; conversion rules
    are identical under WARN and STRICT.

    Set unwrap_result=False to preserve a complete response object, such as
    a contract verdict containing both "verdict" and "result" fields.
    """
    validate_answer_format(answer_format)
    if answer_format in (FRAMED_V1, FRAMED_TEXT_V1):
        return _parse_framed_result(
            text, return_type, unwrap_result=unwrap_result, strict=strict,
            preserve_text=answer_format == FRAMED_TEXT_V1,
        )
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
    answer_format: str = LEGACY,
) -> Any:
    """Return the recovered value, discarding parsing status and diagnostics.

    Use parse_result_detailed() to distinguish valid None from failure.
    """
    return parse_result_detailed(
        text, return_type, strict=strict, answer_format=answer_format,
    ).value
