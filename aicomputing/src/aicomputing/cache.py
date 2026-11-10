"""In-memory caching of predictions with supported built-in values.

Keys preserve value types and dictionary order. Unsupported or cyclic data
bypasses caching. Stored results are copied on insertion and retrieval so
callers cannot mutate the retained prediction through a returned object.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Optional, get_args, get_origin

from .request import ExecuteResult
from .status import ParseStatus, ValidationStatus


def _encode(value: Any, active: set[int]) -> Any:
    """Build a type-tagged key representation for supported built-in values.

    Only known value types are accepted; there is no string fallback for
    arbitrary objects. Type annotations are supported for cache context.
    """
    kind = type(value)
    if kind in (ParseStatus, ValidationStatus):
        return ["status", kind.__name__, value.value]
    if value is None:
        return ["none"]
    if value is Any:
        return ["any-annotation"]
    if value is Ellipsis:
        return ["ellipsis"]
    if kind is bool:
        return ["bool", value]
    if kind is int:
        return ["int", hex(value)]
    if kind is float:
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not cacheable")
        return ["float", value.hex()]
    if kind is str:
        return ["str", value]
    if kind is bytes:
        return ["bytes", value.hex()]

    if kind is type:
        if value in (type(None), bool, int, float, str, bytes, list, tuple, dict, set, frozenset):
            return ["type", value.__name__]
        raise TypeError("custom type annotations are not cacheable")

    origin = get_origin(value)
    if origin is not None:
        return ["annotation", _encode(origin, active), _encode(get_args(value), active)]

    if kind not in (list, tuple, dict, set, frozenset):
        raise TypeError("unsupported cache value")
    identity = id(value)
    if identity in active:
        raise ValueError("cyclic values are not cacheable")
    active.add(identity)
    try:
        if kind is dict:
            # Preserve order and key types, including inside input values.
            return ["dict", [[_encode(k, active), _encode(v, active)] for k, v in value.items()]]
        items = [_encode(item, active) for item in value]
        if kind in (set, frozenset):
            items.sort(key=lambda item: json.dumps(item, separators=(",", ":")))
        return [kind.__name__, items]
    finally:
        active.remove(identity)


def _make_key(
    signature: str,
    docstring: str,
    inputs: Dict[str, Any],
    model: str,
    strategy: str,
    temperature: float,
    top_p: float,
    max_new_tokens: Optional[int],
    context: Any = None,
) -> str | None:
    """Return a key, or None when the request cannot safely be represented."""
    try:
        encoded = _encode(
            (signature, docstring, inputs, model, strategy,
             temperature, top_p, max_new_tokens, context),
            set(),
        )
        blob = json.dumps(encoded, ensure_ascii=True, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError):
        return None
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResultCache:
    """Least-recently-used cache with isolated ExecuteResult snapshots.

    Capacity is an entry count. Zero disables retention. Access is intended
    for a single runtime caller; this cache does not provide locking.
    """

    def __init__(self, maxsize: int = 2048) -> None:
        if type(maxsize) is not int:
            raise TypeError("cache maxsize must be an integer")
        if maxsize < 0:
            raise ValueError("cache maxsize must be non-negative")
        self.maxsize = maxsize
        self._store: OrderedDict[str, ExecuteResult] = OrderedDict()
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
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: Optional[int] = None,
        context: Any = None,
    ) -> ExecuteResult | None:
        if self.maxsize == 0:
            self.misses += 1
            return None
        key = _make_key(
            signature, docstring, inputs, model, strategy,
            temperature, top_p, max_new_tokens, context,
        )
        if key is None or key not in self._store:
            self.misses += 1
            return None
        try:
            result = deepcopy(self._store[key])
        except RecursionError:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return result

    def put(
        self,
        signature: str,
        docstring: str,
        inputs: Dict[str, Any],
        model: str,
        strategy: str,
        result: ExecuteResult,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: Optional[int] = None,
        context: Any = None,
    ) -> None:
        if self.maxsize == 0:
            return
        key = _make_key(
            signature, docstring, inputs, model, strategy,
            temperature, top_p, max_new_tokens, context,
        )
        if key is None or type(result) is not ExecuteResult:
            return
        try:
            # Restrict snapshots to the same known values before copying.
            _encode(vars(result), set())
        except (TypeError, ValueError, RecursionError):
            return
        try:
            snapshot = deepcopy(result)
        except RecursionError:
            return
        self._store[key] = snapshot
        self._store.move_to_end(key)
        while self._store and len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """Remove all entries and reset hit and miss counts."""
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)
