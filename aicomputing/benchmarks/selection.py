"""Deterministic, seeded item and invocation selection.

Pools are deduplicated and sorted. Reordering an unchanged pool preserves
selection; changing pool contents or invocation ordering can change it.
BLAKE2b-derived seeds avoid process-dependent Python string hashes.

Manifests record SELECTION_ALGORITHM_VERSION for compatibility."""

from __future__ import annotations

import hashlib
import random
from typing import List, Sequence

# Increment when selection or fingerprint rules change.
SELECTION_ALGORITHM_VERSION = 4


def _derive_seed(seed: int, *parts: str) -> int:
    """Derive a stable 64-bit sub-seed from NUL-free components.

    NUL separates components. Rejecting it within components prevents
    ambiguous encodings while preserving existing bytes for ordinary inputs.
    """
    components = [str(seed)]
    components.extend(str(part) for part in parts)
    if any("\x00" in component for component in components):
        raise ValueError("seed components must not contain NUL characters")

    payload = "\x00".join(components).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def derive_index(seed: int, item_id: str, n: int, *, salt: str = "invocation") -> int:
    """Select an index using seed, salt, item ID, and positive invocation count.

    Other items do not affect the choice; invocation ordering defines its meaning."""
    if type(n) is not int:
        raise TypeError("n must be an integer")
    if n <= 0:
        raise ValueError(f"n must be positive (item {item_id!r})")
    return _derive_seed(seed, salt, item_id) % n


def select_items(item_ids: Sequence[str], k: int, seed: int,
                 *, salt: str = "items") -> List[str]:
    """Return up to k unique IDs, sorted; k must be a nonnegative integer.

    Sampling is order-independent for unchanged pools. k >= pool size returns all."""
    if type(k) is not int:
        raise TypeError("k must be an integer")
    if k < 0:
        raise ValueError("k must be non-negative")

    pool = sorted(set(item_ids))
    if k >= len(pool):
        return pool
    rng = random.Random(_derive_seed(seed, salt))
    return sorted(rng.sample(pool, k))
