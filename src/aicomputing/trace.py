"""
Tracing module for AI Computing.

Records every LLM call to a JSONL file for debugging and analysis.
Each line is a self-contained JSON object with the full request,
raw model output, parsed result, and diagnostics.

Usage:
    vr = VirtualRuntime(backend="ollama", model="qwen3:4b", trace_file="trace.jsonl")
    # ... execute calls ...
    # trace.jsonl now contains one JSON line per call
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class Tracer:
    """Appends one JSON line per call to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._call_count = 0

    def log(
        self,
        signature: str,
        docstring: str,
        inputs: Dict[str, Any],
        raw_text: str,
        parsed_result: Any,
        model: str,
        strategy: str,
        latency_ms: float,
        tokens_used: Optional[int] = None,
        cached: bool = False,
        retries: int = 0,
        think: Optional[bool] = None,
        type_warnings: Optional[List[str]] = None,
        constraint_warnings: Optional[List[str]] = None,
    ) -> None:
        self._call_count += 1

        entry = {
            "call_id": self._call_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signature": signature,
            "docstring": docstring,
            "inputs": _safe_serialize(inputs),
            "raw_text": raw_text,
            "parsed_result": _safe_serialize(parsed_result),
            "parsed_type": type(parsed_result).__name__,
            "model": model,
            "strategy": strategy,
            "think": think,
            "latency_ms": round(latency_ms, 1),
            "tokens_used": tokens_used,
            "cached": cached,
            "retries": retries,
            "type_warnings": type_warnings or [],
            "constraint_warnings": constraint_warnings or [],
        }

        line = json.dumps(entry, ensure_ascii=False, default=str)

        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    @property
    def call_count(self) -> int:
        return self._call_count


def _safe_serialize(obj: Any) -> Any:
    """Convert an object to something JSON-serializable."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return [_safe_serialize(x) for x in sorted(obj, key=str)]
    if isinstance(obj, bytes):
        return f"<bytes len={len(obj)}>"
    return str(obj)
