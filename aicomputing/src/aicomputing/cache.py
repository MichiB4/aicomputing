"""
Caching layer for AI function calls.

Identical (signature, docstring, inputs, model, strategy) tuples —
plus the generation parameters (think, temperature, top_p,
max_new_tokens) — return cached results without hitting the model
again. This is essential for benchmarks where test harnesses call the
oracle function multiple times with the same arguments.

The generation parameters are part of the key on purpose: without
them, executing the same request with think=True and then think=False
(or at a different temperature) would silently return the first
call's cached result for the second — corrupting exactly the kind of
think-on/off ablation the evaluation is meant to run.
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
    think: Optional[bool],
    temperature: float,
    top_p: float,
    max_new_tokens: Optional[int],
) -> str:
    blob = json.dumps(
        {
            "sig": signature,
            "doc": docstring,
            "in": inputs,
            "model": model,
            "strat": strategy,
            # think=None means "strategy default" — deterministic given the
            # strategy name (already in the key), so None is a valid key part.
            "think": think,
            "temp": temperature,
            "top_p": top_p,
            "max_new": max_new_tokens,
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
        *,
        think: Optional[bool] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: Optional[int] = None,
    ) -> Optional[Any]:
        key = _make_key(signature, docstring, inputs, model, strategy,
                        think, temperature, top_p, max_new_tokens)
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
        *,
        think: Optional[bool] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: Optional[int] = None,
    ) -> None:
        key = _make_key(signature, docstring, inputs, model, strategy,
                        think, temperature, top_p, max_new_tokens)
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
