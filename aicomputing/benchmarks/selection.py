"""
Deterministic selection of benchmark items and invocation indices.

Item pools are deduplicated and sorted before seeded sampling. Reordering an
unchanged pool preserves the selection; adding or removing IDs can change it.
Reproduction requires matching pools, seeds, salts and selection implementation.

Derived seeds use BLAKE2b rather than Python's process-dependent string hash.

select_items() samples from one pool. select_stratified() allocates quotas
across named strata; undersized strata contribute their available items, and
their shortfalls are not redistributed.

derive_index() chooses an invocation index independently for each item.
Other items do not affect that choice, but changing the invocation count or
ordering can change which input is selected.

Experiment manifests record the selection compatibility version, which is
checked by experiment.check_compatible().
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Mapping, Sequence, Tuple

# Compatibility version recorded in experiment manifests and checked by
# experiment.check_compatible(). Increment when reproduction rules change.
# 2: Canonical content fingerprints replaced repr-based fingerprints.
# 3: BBH zero-size requests remain zero; NUL-containing seed components
#    are rejected.
# 4: Typed content fingerprints preserve dictionary order and container types.
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
    """Choose a deterministic index from a positive integer range.

    The choice depends on seed, salt, item_id and n, independently of which
    other items are selected. Invocation ordering determines what the index
    refers to.
    """
    if type(n) is not int:
        raise TypeError("n must be an integer")
    if n <= 0:
        raise ValueError(f"n must be positive (item {item_id!r})")
    return _derive_seed(seed, salt, item_id) % n


def select_items(item_ids: Sequence[str], k: int, seed: int,
                 *, salt: str = "items") -> List[str]:
    """Select up to k unique item IDs, returning them sorted.

    k must be a nonnegative integer. Sampling is order-invariant for an
    unchanged pool. If k reaches or exceeds the number of unique IDs,
    return the entire pool.
    """
    if type(k) is not int:
        raise TypeError("k must be an integer")
    if k < 0:
        raise ValueError("k must be non-negative")

    pool = sorted(set(item_ids))
    if k >= len(pool):
        return pool
    rng = random.Random(_derive_seed(seed, salt))
    return sorted(rng.sample(pool, k))


def select_stratified(strata: Mapping[str, Sequence[str]], total: int, seed: int,
                      *, salt: str = "strata") -> Dict[str, List[str]]:
    """Select IDs using balanced quotas across named strata.

    total must be a nonnegative integer. Assigned quotas differ by at most
    one; a seeded shuffle determines which strata receive the extra slots.

    Each stratum contributes up to its quota. Shortfalls are not redistributed,
    so realized counts may be uneven and substantially below total. Empty
    strata or totals smaller than the stratum count can leave strata unselected.
    """
    if type(total) is not int:
        raise TypeError("total must be an integer")
    if total < 0:
        raise ValueError("total must be non-negative")

    names = sorted(strata)
    if not names:
        return {}

    base, remainder = divmod(total, len(names))
    order = list(names)
    random.Random(_derive_seed(seed, salt, "remainder")).shuffle(order)
    extra = set(order[:remainder])

    out: Dict[str, List[str]] = {}
    for name in names:
        quota = base + (1 if name in extra else 0)
        out[name] = select_items(strata[name], quota, seed, salt=f"{salt}:{name}")
    return out


def flatten_stratified(selection: Mapping[str, Sequence[str]]) -> List[Tuple[str, str]]:
    """Flatten a stratified selection into sorted (stratum, item_id) pairs."""
    return [
        (stratum, item_id)
        for stratum in sorted(selection)
        for item_id in sorted(selection[stratum])
    ]
