"""Media value wrappers and file/base64 factories."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Union


# Multimodal wrapper types.

class Modality(str, Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"


@dataclass
class MediaObject:
    """Media bytes with a modality, MIME type, and optional metadata."""

    modality: Modality
    data: bytes = field(repr=False)
    mime_type: str = "application/octet-stream"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Union[str, Path], **kwargs) -> MediaObject:
        p = Path(path)
        data = p.read_bytes()
        mime = _guess_mime(p.suffix)
        modality = _suffix_to_modality(p.suffix)
        return cls(modality=modality, data=data, mime_type=mime, **kwargs)

    @classmethod
    def from_base64(
        cls,
        b64: str,
        modality: Modality,
        mime_type: str,
    ) -> MediaObject:
        return cls(
            modality=modality,
            data=base64.b64decode(b64),
            mime_type=mime_type,
        )

    def to_base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def __len__(self) -> int:
        """Return the number of stored bytes."""
        return len(self.data)


class Image(MediaObject):
    """Image data wrapper."""

    def __init__(self, data: bytes, mime_type: str = "image/png", **kw) -> None:
        super().__init__(
            modality=Modality.IMAGE,
            data=data,
            mime_type=mime_type,
            **kw,
        )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> Image:
        p = Path(path)
        return cls(data=p.read_bytes(), mime_type=_guess_mime(p.suffix))

    @classmethod
    def from_base64(
        cls,
        b64: str,
        modality: Modality,
        mime_type: str,
    ) -> Image:
        if modality != Modality.IMAGE:
            raise ValueError("Image.from_base64 requires image modality")
        return cls(data=base64.b64decode(b64), mime_type=mime_type)


class Audio(MediaObject):
    """Audio data wrapper."""

    def __init__(self, data: bytes, mime_type: str = "audio/wav", **kw) -> None:
        super().__init__(
            modality=Modality.AUDIO,
            data=data,
            mime_type=mime_type,
            **kw,
        )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> Audio:
        p = Path(path)
        return cls(data=p.read_bytes(), mime_type=_guess_mime(p.suffix))

    @classmethod
    def from_base64(
        cls,
        b64: str,
        modality: Modality,
        mime_type: str,
    ) -> Audio:
        if modality != Modality.AUDIO:
            raise ValueError("Audio.from_base64 requires audio modality")
        return cls(data=base64.b64decode(b64), mime_type=mime_type)


class Video(MediaObject):
    """Video data wrapper."""

    def __init__(self, data: bytes, mime_type: str = "video/mp4", **kw) -> None:
        super().__init__(
            modality=Modality.VIDEO,
            data=data,
            mime_type=mime_type,
            **kw,
        )

    @classmethod
    def from_file(cls, path: Union[str, Path], **kwargs) -> Video:
        p = Path(path)
        return cls(
            data=p.read_bytes(),
            mime_type=_guess_mime(p.suffix),
            **kwargs,
        )

    @classmethod
    def from_base64(
        cls,
        b64: str,
        modality: Modality,
        mime_type: str,
    ) -> Video:
        if modality != Modality.VIDEO:
            raise ValueError("Video.from_base64 requires video modality")
        return cls(data=base64.b64decode(b64), mime_type=mime_type)


class Text(MediaObject):
    """UTF-8 text wrapper."""

    def __init__(
        self,
        content: str,
        mime_type: str = "text/plain",
        **kw,
    ) -> None:
        super().__init__(
            modality=Modality.TEXT,
            data=content.encode("utf-8"),
            mime_type=mime_type,
            **kw,
        )

    @classmethod
    def from_file(cls, path: Union[str, Path], **kwargs) -> Text:
        p = Path(path)
        return cls(
            content=p.read_bytes().decode("utf-8"),
            mime_type=_guess_mime(p.suffix),
            **kwargs,
        )

    @classmethod
    def from_base64(
        cls,
        b64: str,
        modality: Modality,
        mime_type: str,
    ) -> Text:
        if modality != Modality.TEXT:
            raise ValueError("Text.from_base64 requires text modality")
        return cls(
            content=base64.b64decode(b64).decode("utf-8"),
            mime_type=mime_type,
        )

    @property
    def content(self) -> str:
        return self.data.decode("utf-8")


# MIME helpers.

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
    """Return a suffix-based MIME label, or the generic binary fallback."""
    return _MIME_MAP.get(suffix.lower(), "application/octet-stream")


def _suffix_to_modality(suffix: str) -> Modality:
    """Return a suffix-based modality; unknown suffixes currently use TEXT."""
    return _MODALITY_MAP.get(suffix.lower(), Modality.TEXT)


# Retain public class paths for diagnostics and serialized values.
for _public_type in (Modality, MediaObject, Image, Audio, Video, Text):
    _public_type.__module__ = "aicomputing.types"
del _public_type
