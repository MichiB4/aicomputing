"""Shared benchmark metadata, task specifications and manifest access."""

from __future__ import annotations

from typing import Any, Dict, List

CORE_BENCHMARKS = ("humaneval_plus", "mbpp_plus")
GENERALIZATION_BENCHMARKS = ("gsm8k",)
ALL_BENCHMARKS = CORE_BENCHMARKS + GENERALIZATION_BENCHMARKS

BENCHMARK_GROUP: Dict[str, str] = {
    **{b: "core" for b in CORE_BENCHMARKS},
    **{b: "generalization" for b in GENERALIZATION_BENCHMARKS},
}


def manifest_items(manifest: Dict[str, Any], benchmark: str) -> List[Dict[str, Any]]:
    entry = manifest["benchmarks"].get(benchmark)
    if entry is None:
        raise ValueError(
            f"benchmark {benchmark!r} is not in this manifest "
            f"(has: {sorted(manifest['benchmarks'])})"
        )
    return entry["items"]
