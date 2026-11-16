"""Select reviewed tasks and resolve them into specifications, inputs and targets.

Models receive only specifications and inputs; expected values remain grading
data. Pool fingerprints record the approved item IDs; content fingerprints
cover each selected specification, input and target.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from aicomputing.types import parse_signature

from . import BENCHMARK_GROUP
from .prepared_data import approved_items, load_release
from .selection import SELECTION_ALGORITHM_VERSION, derive_index, select_items
from .values import canonical_json, decode_value


@dataclass
class ResolvedItem:
    """One selected invocation, its specification, and grading/provenance metadata."""
    item_id: str
    signature: str
    docstring: str
    inputs: Dict[str, Any]
    expected: Any
    scoring_profile: str
    invocation_id: Optional[str] = None
    input_line: str = ""
    invocation_pool: Optional[str] = None  # Source label, e.g. base or plus; None if unavailable.
    return_type_source: str = "signature"    # "signature" | "none"


def select_tasks(release_path, benchmarks: Sequence[str], samples: int, seed: int):
    """Return the selection record and the resolved items for each benchmark.

    Every benchmark contributes the same seeded prefix of its approved pool,
    with one seeded invocation per task.
    """
    data, reference = load_release(release_path)
    record: Dict[str, Any] = {"dataset": reference, "algorithm_version": SELECTION_ALGORITHM_VERSION,
                              "seed": seed, "benchmarks": {}}
    resolved: Dict[str, List[ResolvedItem]] = {}
    for benchmark in benchmarks:
        pool = {task["item_id"]: task for task in approved_items(data, benchmark)}
        if samples > len(pool):
            raise ValueError(f"{benchmark}: requested {samples} tasks, but only {len(pool)} are approved; "
                             "finish review before sampling")
        items = [_resolve(pool[item_id], seed)
                 for item_id in select_items(list(pool), samples, seed, salt=benchmark)]
        entries = []
        for item in items:
            entry = {"item_id": item.item_id, "invocation_id": item.invocation_id,
                     "content_fingerprint": content_fingerprint(item)}
            if item.invocation_pool:
                entry["invocation_pool"] = item.invocation_pool
            entries.append(entry)
        record["benchmarks"][benchmark] = {
            "group": BENCHMARK_GROUP[benchmark],
            "display_name": benchmark,
            "requested": samples,
            "selected": len(entries),
            "pool_size": len(pool),
            "pool_fingerprint": _fingerprint(sorted(pool)),
            "dataset_revision": reference["sha256"],
            "invocation_pool_policy": _describe_pool_policy(entries),
            "items": entries,
            "content_fingerprints": True,
        }
        resolved[benchmark] = items
    return record, resolved


def _resolve(task, seed: int) -> ResolvedItem:
    """Choose one invocation per task independently of sample size."""
    observations = task["observations"]
    observation = observations[derive_index(seed, task["item_id"], len(observations))]
    annotation = parse_signature(task["signature"]).return_annotation
    return ResolvedItem(
        item_id=task["item_id"], signature=task["signature"], docstring=task["specification"],
        inputs=decode_value(observation["inputs"]), expected=decode_value(observation["expected"]),
        scoring_profile=task["scoring_profile"], invocation_id=observation["invocation_id"],
        input_line=observation.get("input_line", ""), invocation_pool=observation.get("pool"),
        return_type_source="signature" if annotation is not None else "none",
    )


def content_fingerprint(item: ResolvedItem) -> str:
    """Hash the specification, inputs and expected output."""
    payload = canonical_json([item.signature, item.docstring, item.inputs, item.expected])
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def _fingerprint(values: Sequence[str]) -> str:
    """Hash an ordered ID sequence; callers sort when order is irrelevant."""
    payload = canonical_json(list(values))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def _describe_pool_policy(items: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Summarize selected invocation labels without inferring cache availability.

    No labels means source metadata is unavailable. When only some selected
    items have labels, include unknown rather than hide the missing metadata.
    """
    pools = {item.get("invocation_pool") for item in items if item.get("invocation_pool")}
    if not pools:
        return None
    if any(not item.get("invocation_pool") for item in items):
        pools.add("unknown")
    if pools == {"base"}:
        return "base_only"
    if pools == {"plus"}:
        return "plus_only"
    return "+".join(sorted(pools))
