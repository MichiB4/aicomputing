"""Append invocation evidence as JSONL; write failures warn without stopping predictions.

Cache hits record cached=true and zero tokens. Token counts for uncached calls
accumulate returned responses across stages and validation retries. Reasoning requests
preserve response order but do not establish the model's actual reasoning mode.

Enable with VirtualRuntime(backend="ollama", trace_file="trace.jsonl").
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .request import ExecuteRequest, ExecuteResult


logger = logging.getLogger("aicomputing")
_USE_RESULT = object()


class Tracer:
    """Append JSONL records; warn if serialization or writing fails."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._call_count = 0

    def log(
        self,
        request: ExecuteRequest,
        result: ExecuteResult,
        *,
        parsed_result: Any = _USE_RESULT,
    ) -> None:
        """Record a result, optionally retaining a candidate withheld by validation."""
        # Protect record numbering and writing through this tracer instance.
        with self._lock:
            self._call_count += 1

            try:
                parsed = result.result if parsed_result is _USE_RESULT else parsed_result
                entry = {
                    "call_id": self._call_count,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "signature": request.signature,
                    "docstring": request.docstring,
                    "inputs": request.inputs,
                    # Only declared result fields; no copying of arbitrary caller objects.
                    **{field.name: getattr(result, field.name) for field in fields(ExecuteResult)},
                    "parsed_result": parsed,
                    "parsed_type": type(parsed).__name__,
                }
                entry["latency_ms"] = None if result.latency_ms is None else round(result.latency_ms, 1)

                line = json.dumps(_safe_serialize(entry), ensure_ascii=False, default=str)

                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8", errors="backslashreplace") as f:
                    f.write(line + "\n")
            except Exception:
                # Optional diagnostics must not prevent returning a prediction.
                logger.warning(
                    "Could not record trace to %s",
                    self.path,
                    exc_info=True,
                )

    @property
    def call_count(self) -> int:
        """Number of log attempts, including failed serialization or writes."""
        with self._lock:
            return self._call_count


def _safe_serialize(obj: Any) -> Any:
    """Convert values to diagnostic representations for JSON encoding."""
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
