"""
Caching layer for AI function calls.

Identical (signature, docstring, inputs, model, strategy) tuples
return cached results without hitting the model again. This is
essential for benchmarks where test harnesses call the oracle function
multiple times with the same arguments.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any, Dict, Optional


def _make_key(
    signature: str,
    docstring: str,
    inputs: Dict[str, Any],
    model: str,
    strategy: str,
) -> str:
    blob = json.dumps(
        {
            "sig": signature,
            "doc": docstring,
            "in": inputs,
            "model": model,
            "strat": strategy,
        },
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class ResultCache:
    """Simple in-memory LRU cache for ExecuteResult objects."""

    def __init__(self, maxsize: int = 2048) -> None:
        self.maxsize = maxsize
        self._store: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(
        self,
        signature: str,
        docstring: str,
        inputs: Dict[str, Any],
        model: str,
        strategy: str,
    ) -> Optional[Any]:
        key = _make_key(signature, docstring, inputs, model, strategy)
        if key in self._store:
            self.hits += 1
            self._store.move_to_end(key)
            return self._store[key]
        self.misses += 1
        return None

    def put(
        self,
        signature: str,
        docstring: str,
        inputs: Dict[str, Any],
        model: str,
        strategy: str,
        result: Any,
    ) -> None:
        key = _make_key(signature, docstring, inputs, model, strategy)
        self._store[key] = result
        self._store.move_to_end(key)
        if len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)
