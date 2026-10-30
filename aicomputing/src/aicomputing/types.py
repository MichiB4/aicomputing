"""Function-signature parsing and public media type imports for AI Computing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, get_args, get_origin

from .media import (
    Audio, Image, MediaObject, Modality, Text, Video,
)


# Signature parsing.

_TYPE_ALIASES: Dict[str, Any] = {
    "int": int, "float": float, "str": str, "bool": bool,
    "None": type(None), "none": type(None),
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "List": list, "Dict": dict, "Tuple": tuple, "Set": set,
    "Any": Any,
    "Image": Image, "Audio": Audio, "Video": Video, "Text": Text,
    "MediaObject": MediaObject,
}

# Nested parentheses are unsupported outside quoted defaults.
_SIG_RE = re.compile(
    r"^\s*(?P<name>\w+)\s*\("
    r"(?P<params>[^)]*)"
    r"\)\s*(?:->\s*(?P<ret>.+))?\s*$"
)


def _parse_param(part: str) -> Optional[ParsedParam]:
    """Parse a parameter fragment, preserving its default as source text."""
    part = part.strip()
    if not part:
        return None

    # Find a top-level '=' using bracket depth; quoted text is not interpreted.
    default: Optional[str] = None
    depth = 0
    for i, ch in enumerate(part):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "=" and depth == 0:
            default = part[i + 1:].strip() or None
            part = part[:i].strip()
            break

    # Split the parameter name from its annotation.
    if ":" in part:
        pname, ptype_str = part.split(":", 1)
        pname = pname.strip()
        ptype_str = ptype_str.strip() or None
    else:
        pname, ptype_str = part.strip(), None

    if not pname or not re.match(r"^\*{0,2}\w+$", pname):
        return None

    if pname.startswith("*"):
        raise UnsupportedAnnotationError(
            f"Variadic parameters are unsupported: {pname!r}"
        )

    resolved = resolve_type_str(ptype_str) if ptype_str else Any
    return ParsedParam(
        name=pname,
        annotation=ptype_str,
        resolved_type=resolved,
        default=default,
    )


@dataclass
class ParsedParam:
    """Parameter name, annotation, resolved type, and default source text."""

    name: str
    annotation: Optional[str] = None
    resolved_type: Any = None

    # None means absent; an explicit None default is stored as "None".
    default: Optional[str] = None


@dataclass
class ParsedSignature:
    """Function name, ordered parameters, and return-type information."""

    name: str
    params: List[ParsedParam]
    return_annotation: Optional[str] = None
    return_type: Any = None


class UnsupportedAnnotationError(ValueError):
    """A specification has unsupported annotations or malformed quoted text."""


def validate_type_annotation(annotation: Any) -> None:
    """Reject unsupported annotations before values or model output are checked."""
    if annotation is Any or annotation is None:
        return

    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type):
            # Some typing classes (e.g. TypedDict) reject isinstance checks.
            try:
                isinstance(None, annotation)
            except TypeError as exc:
                raise UnsupportedAnnotationError(
                    f"Unsupported type annotation: {annotation!r}"
                ) from exc
            return
        raise UnsupportedAnnotationError(
            f"Unsupported type annotation: {annotation!r}"
        )

    if origin not in (list, dict, tuple, set):
        raise UnsupportedAnnotationError(
            f"Unsupported type annotation: {annotation!r}"
        )

    args = get_args(annotation)
    if not args:
        # Bare typing aliases are supported; explicit empty generic arguments
        # (including tuple[()]) are not currently checked by the validator.
        if annotation not in (List, Dict, Tuple, Set):
            raise UnsupportedAnnotationError(
                f"Unsupported empty generic annotation: {annotation!r}"
            )
        return

    if origin in (list, set) and len(args) != 1:
        raise UnsupportedAnnotationError(
            f"Expected one type argument in {annotation!r}"
        )
    if origin is dict and len(args) != 2:
        raise UnsupportedAnnotationError(
            f"Expected two type arguments in {annotation!r}"
        )
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
        validate_type_annotation(args[0])
        return

    for arg in args:
        validate_type_annotation(arg)


def resolve_type_str(s: str) -> Any:
    """Resolve supported annotation text; raise for unknown or invalid forms."""
    s = s.strip()
    if s in _TYPE_ALIASES:
        return _TYPE_ALIASES[s]

    match = re.fullmatch(r"(\w+)\[(.+)\]", s)
    if match:
        origin_name, args_text = match.groups()
        origin = _TYPE_ALIASES.get(origin_name)
        if origin not in (list, dict, tuple, set):
            raise UnsupportedAnnotationError(f"Unsupported type annotation: {s!r}")

        parts = _split_type_args(args_text)
        if not parts or any(not part for part in parts):
            raise UnsupportedAnnotationError(f"Invalid type annotation: {s!r}")
        resolved = tuple(
            Ellipsis
            if origin is tuple and len(parts) == 2 and index == 1 and part == "..."
            else resolve_type_str(part)
            for index, part in enumerate(parts)
        )
        try:
            annotation = origin[resolved if len(resolved) > 1 else resolved[0]]
        except TypeError as exc:
            raise UnsupportedAnnotationError(
                f"Invalid type annotation: {s!r}"
            ) from exc
        validate_type_annotation(annotation)
        return annotation

    raise UnsupportedAnnotationError(f"Unsupported type annotation: {s!r}")


def _mask_string_contents(s: str) -> str:
    """Mask quoted contents while preserving offsets and quote delimiters."""
    masked = list(s)
    quote: Optional[str] = None
    escaped = False
    index = 0
    while index < len(s):
        ch = s[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif s.startswith(quote, index):
                index += len(quote)
                quote = None
                continue
            if ch not in "\r\n":
                masked[index] = " "
        elif ch in ("'", '"'):
            quote = ch * 3 if s.startswith(ch * 3, index) else ch
            index += len(quote)
            continue
        index += 1
    if quote is not None:
        raise UnsupportedAnnotationError(f"Unterminated quoted text: {s!r}")
    return "".join(masked)


def _split_type_args(s: str) -> List[str]:
    """Split at unquoted, top-level commas while preserving source text."""
    parts: List[str] = []
    depth = 0
    current: List[str] = []

    for ch, masked_ch in zip(s, _mask_string_contents(s)):
        if masked_ch in ("(", "[", "{"):
            depth += 1
            current.append(ch)
        elif masked_ch in (")", "]", "}"):
            depth -= 1
            current.append(ch)
        elif masked_ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def parse_signature(sig: str) -> ParsedSignature:
    """Parse supported signature text into names and type information."""
    m = _SIG_RE.match(_mask_string_contents(sig))
    if not m:
        raise ValueError(f"Cannot parse signature: {sig!r}")

    name = sig[m.start("name"):m.end("name")]
    raw_params = sig[m.start("params"):m.end("params")].strip()
    return_annotation = (
        sig[m.start("ret"):m.end("ret")].strip() if m.group("ret") else None
    )

    params: List[ParsedParam] = []
    if raw_params:
        # Preserve nested generic arguments while splitting parameters.
        for part in _split_type_args(raw_params):
            parsed = _parse_param(part)
            if parsed is not None:
                params.append(parsed)

    ret_type = (
        resolve_type_str(return_annotation)
        if return_annotation is not None
        else Any
    )

    return ParsedSignature(
        name=name,
        params=params,
        return_annotation=return_annotation,
        return_type=ret_type,
    )
