"""
Deterministic, reproducible benchmark selection.

Every sampling decision in the experiment goes through this module so that a
rerun with the same seed evaluates exactly the same items, and so that each
model and strategy sees an identical item set (the precondition for the paired
comparisons in the analysis plan).

Two properties matter and neither is automatic:

Order invariance
    Selection is derived from item IDs, not from list positions. A dataset
    that ships its rows in a different order on a later `datasets` version,
    or a HuggingFace revision that inserts a row, would otherwise silently
    change the sample while the seed stayed the same. IDs are sorted before
    any sampling, so the pool is canonical regardless of load order.

Stable hashing
    Python's built-in hash() is randomised per process for str/bytes
    (PYTHONHASHSEED), so it cannot be used for anything reproducible across
    runs. Every derived seed here goes through hashlib.blake2b instead.

The two entry points differ in what they guarantee:

    select_items()       k items from one pool, uniformly at random.
    select_stratified()  k items spread evenly across named strata, so a
                         sample cannot be dominated by whichever stratum
                         happens to be largest or first (BBH, requirement 4).

For choosing one invocation *within* an item, use derive_index(), which gives
each item an independent draw. That is deliberately not "take the first": the
EvalPlus suites are ordered with the original hand-written cases first and the
generated stress cases after, so index 0 is systematically the easiest input
for every function in the benchmark.
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Mapping, Sequence, Tuple

# Bumped only when a change here would alter which items a given seed selects.
# The manifest records it, so a stale manifest is detectable rather than
# silently reproducing a different sample. See manifest.check_compatible().
# 2: content fingerprints changed from repr() to a canonical recursive
#    serialisation, which changes manifest identity semantics even though the
#    SELECTED items are unchanged. Bumped so an older manifest is refused
#    rather than silently re-verified under different rules.
SELECTION_ALGORITHM_VERSION = 2


def _derive_seed(seed: int, *parts: str) -> int:
    """
    Derive a stable 64-bit sub-seed from a base seed and string parts.

    Uses blake2b rather than hash() because the built-in is salted per process
    for strings — a seeded run would otherwise select different items on every
    invocation, which is the exact failure this module exists to prevent.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    for p in parts:
        h.update(b"\x00")
        h.update(str(p).encode("utf-8"))
    return int.from_bytes(h.digest(), "big")


def derive_index(seed: int, item_id: str, n: int, *, salt: str = "invocation") -> int:
    """
    Pick one index in range(n) for a single item, deterministically.

    Each item draws independently from its own derived seed, so adding or
    removing items does not shift the choice made for any other item. That
    keeps a manifest stable under a change of sample size, which matters
    because the pilot and the main experiment select different numbers of
    items from the same pools and should agree on the invocation chosen for
    any item they share.
    """
    if n <= 0:
        raise ValueError(f"cannot choose an index from an empty range (item {item_id!r})")
    return _derive_seed(seed, salt, item_id) % n


def select_items(item_ids: Sequence[str], k: int, seed: int,
                 *, salt: str = "items") -> List[str]:
    """
    Select k item IDs from a pool, deterministically and order-invariantly.

    Returns them in sorted order. If k >= len(pool) the whole pool is
    returned — asking for more than exists is normal here (HumanEval+ takes
    every eligible function) and is not an error.
    """
    pool = sorted(set(item_ids))
    if k >= len(pool):
        return pool
    rng = random.Random(_derive_seed(seed, salt))
    return sorted(rng.sample(pool, k))


def select_stratified(strata: Mapping[str, Sequence[str]], total: int, seed: int,
                      *, salt: str = "strata") -> Dict[str, List[str]]:
    """
    Select ~`total` IDs spread as evenly as possible across named strata.

    Requirement 4: BBH must not be sampled by a pooled prefix. Concatenating
    the task splits and taking the first N gives a sample made almost entirely
    of whichever task sorts first, so the "BBH accuracy" number would really be
    that one task's accuracy. Allocating per task and sampling within each task
    keeps every category represented.

    Allocation is floor(total / n_strata) each, with the remainder handed out
    in a deterministic order derived from the seed rather than to whichever
    strata sort first — otherwise the same one or two categories would get the
    extra item in every experiment. A stratum smaller than its allocation
    contributes everything it has; the shortfall is NOT redistributed, so the
    per-stratum count stays balanced and the total may fall slightly below
    `total`. The manifest records the realised counts.
    """
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
    return [(s, i) for s in sorted(selection) for i in sorted(selection[s])]
