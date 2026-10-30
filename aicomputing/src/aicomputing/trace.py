"""Append invocation evidence as JSONL; write failures warn without stopping predictions.

Cache hits record cached=true and zero tokens. Token counts for uncached calls
use legacy strategy accounting and may omit earlier attempts. Reasoning requests
preserve response order but do not establish the model's actual reasoning mode.

Enable with VirtualRuntime(trace_file="trace.jsonl").
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger("aicomputing")


class Tracer:
    """Append JSONL records; warn if serialization or writing fails."""

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
        token_budget_used: Optional[int] = None,
        cached: bool = False,
        retries: int = 0,
        reasoning_requests: Optional[List[Optional[Dict[str, Any]]]] = None,
        substrategy: Optional[str] = None,
        revised: Optional[bool] = None,
        pre_revision_value: Any = None,
        provider_finish_reason: Optional[str] = None,
        prompts: Optional[List[str]] = None,
        input_warnings: Optional[List[str]] = None,
        type_warnings: Optional[List[str]] = None,
        constraint_warnings: Optional[List[str]] = None,
        reasoning_observations: Optional[List[Optional[Dict[str, Any]]]] = None,
        answer_format: str = "legacy",
        transport_diagnostics: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> None:
        # Protect record numbering and writing through this tracer instance.
        with self._lock:
            self._call_count += 1

            try:
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
                    "substrategy": substrategy,
                    "revised": revised,
                    "pre_revision_value": _safe_serialize(pre_revision_value),
                    "provider_finish_reason": provider_finish_reason,
                    # Strategy-recorded prompts supplied by the runtime.
                    "prompts": prompts or [],
                    "reasoning_requests": _safe_serialize(reasoning_requests),
                    "reasoning_observations": _safe_serialize(reasoning_observations),
                    "answer_format": answer_format,
                    "transport_diagnostics": _safe_serialize(transport_diagnostics),
                    "latency_ms": round(latency_ms, 1),
                    # Legacy strategy-reported token count.
                    "tokens_used": tokens_used,
                    # Strategy-reported output-token ceiling, not tokens consumed.
                    "token_budget_used": token_budget_used,
                    "cached": cached,
                    "retries": retries,
                    "input_warnings": input_warnings or [],
                    "type_warnings": type_warnings or [],
                    "constraint_warnings": constraint_warnings or [],
                }

                line = json.dumps(entry, ensure_ascii=False, default=str)

                with open(self.path, "a", encoding="utf-8") as f:
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
