"""
Extended type system for AI Computing.

Covers both standard Python types and multimodal types (Image, Audio, Video, Text).
Also provides signature parsing to extract input/output types for runtime type-checking.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Multimodal wrapper types
# ---------------------------------------------------------------------------

class Modality(str, Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"


@dataclass
class MediaObject:
    """Base for all multimodal data passed through AI functions."""
    modality: Modality
    data: bytes = field(repr=False)
    mime_type: str = "application/octet-stream"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Union[str, Path], **kwargs) -> "MediaObject":
        p = Path(path)
        data = p.read_bytes()
        mime = _guess_mime(p.suffix)
        modality = _suffix_to_modality(p.suffix)
        return cls(modality=modality, data=data, mime_type=mime, **kwargs)

    @classmethod
    def from_base64(cls, b64: str, modality: Modality, mime_type: str) -> "MediaObject":
        return cls(modality=modality, data=base64.b64decode(b64), mime_type=mime_type)

    def to_base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def __len__(self) -> int:
        return len(self.data)


class Image(MediaObject):
    """Image data wrapper."""
    def __init__(self, data: bytes, mime_type: str = "image/png", **kw):
        super().__init__(modality=Modality.IMAGE, data=data, mime_type=mime_type, **kw)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Image":
        p = Path(path)
        return cls(data=p.read_bytes(), mime_type=_guess_mime(p.suffix))


class Audio(MediaObject):
    """Audio data wrapper."""
    def __init__(self, data: bytes, mime_type: str = "audio/wav", **kw):
        super().__init__(modality=Modality.AUDIO, data=data, mime_type=mime_type, **kw)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Audio":
        p = Path(path)
        return cls(data=p.read_bytes(), mime_type=_guess_mime(p.suffix))


class Video(MediaObject):
    """Video data wrapper."""
    def __init__(self, data: bytes, mime_type: str = "video/mp4", **kw):
        super().__init__(modality=Modality.VIDEO, data=data, mime_type=mime_type, **kw)


class Text(MediaObject):
    """Rich text wrapper (for long documents, markdown, etc.)."""
    def __init__(self, content: str, mime_type: str = "text/plain", **kw):
        super().__init__(
            modality=Modality.TEXT,
            data=content.encode("utf-8"),
            mime_type=mime_type,
            **kw,
        )

    @property
    def content(self) -> str:
        return self.data.decode("utf-8")


# ---------------------------------------------------------------------------
# MIME helpers
# ---------------------------------------------------------------------------

_MIME_MAP = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".mp4": "video/mp4", ".webm": "video/webm", ".avi": "video/x-msvideo",
    ".txt": "text/plain", ".md": "text/markdown", ".html": "text/html",
}

_MODALITY_MAP = {
    ".png": Modality.IMAGE, ".jpg": Modality.IMAGE, ".jpeg": Modality.IMAGE,
    ".gif": Modality.IMAGE, ".webp": Modality.IMAGE, ".bmp": Modality.IMAGE,
    ".svg": Modality.IMAGE,
    ".wav": Modality.AUDIO, ".mp3": Modality.AUDIO, ".flac": Modality.AUDIO,
    ".ogg": Modality.AUDIO,
    ".mp4": Modality.VIDEO, ".webm": Modality.VIDEO, ".avi": Modality.VIDEO,
    ".txt": Modality.TEXT, ".md": Modality.TEXT, ".html": Modality.TEXT,
}


def _guess_mime(suffix: str) -> str:
    return _MIME_MAP.get(suffix.lower(), "application/octet-stream")


def _suffix_to_modality(suffix: str) -> Modality:
    return _MODALITY_MAP.get(suffix.lower(), Modality.TEXT)


# ---------------------------------------------------------------------------
# Signature parsing
# ---------------------------------------------------------------------------

# Mapping from type-annotation strings to Python types.
_TYPE_ALIASES: Dict[str, Any] = {
    "int": int, "float": float, "str": str, "bool": bool,
    "None": type(None), "none": type(None),
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "List": list, "Dict": dict, "Tuple": tuple, "Set": set,
    "Any": Any,
    "Image": Image, "Audio": Audio, "Video": Video, "Text": Text,
    "MediaObject": MediaObject,
}

# Regex for a function signature like  func_name(arg1: type1, arg2: type2) -> return_type
_SIG_RE = re.compile(
    r"^\s*(?P<name>\w+)\s*\("
    r"(?P<params>[^)]*)"
    r"\)\s*(?:->\s*(?P<ret>.+))?\s*$"
)

def _parse_param(part: str) -> Optional["ParsedParam"]:
    """Parse a single parameter string like 'x', 'x: int', or 'x: int = 3'."""
    part = part.strip()
    if not part:
        return None

    # Split off the default value at the first top-level '='
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

    # Split name from annotation at the first ':'
    if ":" in part:
        pname, ptype_str = part.split(":", 1)
        pname = pname.strip()
        ptype_str = ptype_str.strip() or None
    else:
        pname, ptype_str = part.strip(), None

    if not pname or not re.match(r"^\*{0,2}\w+$", pname):
        return None

    resolved = resolve_type_str(ptype_str) if ptype_str else Any
    return ParsedParam(
        name=pname.lstrip("*"),
        annotation=ptype_str,
        resolved_type=resolved,
        default=default,
    )


@dataclass
class ParsedParam:
    name: str
    annotation: Optional[str] = None
    resolved_type: Any = None
    default: Optional[str] = None


@dataclass
class ParsedSignature:
    name: str
    params: List[ParsedParam]
    return_annotation: Optional[str] = None
    return_type: Any = None


def resolve_type_str(s: str) -> Any:
    """Resolve a type annotation string to a Python type (best effort)."""
    s = s.strip()
    if s in _TYPE_ALIASES:
        return _TYPE_ALIASES[s]

    # Handle generic forms like list[int], dict[str, int], etc.
    m = re.match(r"^(\w+)\[(.+)\]$", s)
    if m:
        origin_str, args_str = m.group(1), m.group(2)
        origin = _TYPE_ALIASES.get(origin_str)
        if origin is None:
            return Any
        # Split args on top-level commas (respecting brackets)
        args = _split_type_args(args_str)
        resolved_args = tuple(resolve_type_str(a) for a in args)
        try:
            return origin[resolved_args] if len(resolved_args) > 1 else origin[resolved_args[0]]
        except TypeError:
            return origin
    return Any


def _split_type_args(s: str) -> List[str]:
    """Split 'str, int' or 'str, list[int]' respecting bracket depth."""
    parts, depth, current = [], 0, []
    for ch in s:
        if ch in ("(", "[", "{"):
            depth += 1
            current.append(ch)
        elif ch in (")", "]", "}"):
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def parse_signature(sig: str) -> ParsedSignature:
    """Parse a function signature string into structured form."""
    m = _SIG_RE.match(sig)
    if not m:
        raise ValueError(f"Cannot parse signature: {sig!r}")

    name = m.group("name")
    raw_params = m.group("params").strip()
    raw_ret = m.group("ret")

    params: List[ParsedParam] = []
    if raw_params:
        # Split on top-level commas only, so generic annotations like
        # dict[str, int] or list[tuple[int, int]] stay intact.
        for part in _split_type_args(raw_params):
            parsed = _parse_param(part)
            if parsed is not None:
                params.append(parsed)

    ret_type = resolve_type_str(raw_ret.strip()) if raw_ret else Any

    return ParsedSignature(
        name=name,
        params=params,
        return_annotation=raw_ret.strip() if raw_ret else None,
        return_type=ret_type,
    )
