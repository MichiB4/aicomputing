"""Resolve manifest selections into specifications, inputs, and grading targets.

Models receive only specifications and inputs. Ground truth remains grading data.
Shared comparison functions do not guarantee historical comparability: parsing,
validation, and failure handling also affect end-to-end scores.
Content fingerprints cover the selected specification, input, and target."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from aicomputing.types import parse_signature

from benchmark_metadata import BENCHMARK_GROUP, manifest_items


@dataclass
class ResolvedItem:
    """One selected invocation, its specification, and grading/provenance metadata."""
    item_id: str
    signature: str
    docstring: str
    inputs: Dict[str, Any]
    expected: Any
    return_type: Optional[type] = None  # None leaves type inference to the runtime.
    category: Optional[str] = None
    invocation_id: Optional[str] = None
    input_line: str = ""
    invocation_pool: Optional[str] = None  # Source label, e.g. base or plus; None if unavailable.
    return_type_source: str = "signature"    # "signature" | "none"
    scoring_profile: Optional[str] = None  # Frozen prepared-release grading rule.


@dataclass
class Adapter:
    """Resolve a benchmark selection and compare predictions with expected values."""
    name: str
    resolve: Callable[[Dict[str, Any]], List[ResolvedItem]]
    compare: Callable[[Any, Any], bool]

    @property
    def group(self) -> str:
        return BENCHMARK_GROUP[self.name]

    def score(self, predicted, item: ResolvedItem) -> bool:
        if item.scoring_profile is not None:
            from prepared_data import compare_prepared
            return compare_prepared(predicted, item.expected, item.scoring_profile)
        return self.compare(predicted, item.expected)


def _resolve_prepared(manifest, benchmark):
    from mbpp_ground_truth import decode_value
    from prepared_data import approved_items, release_for_manifest
    data = release_for_manifest(manifest)
    available = {item["item_id"]: item for item in approved_items(data, benchmark)}
    out, seen = [], set()
    for entry in manifest_items(manifest, benchmark):
        item_id = entry["item_id"]
        if item_id in seen:
            raise ValueError(f"{benchmark}: duplicate selected item ID {item_id}")
        seen.add(item_id)
        if item_id not in available:
            raise ValueError(f"{item_id}: task is not approved in the frozen release")
        task = available[item_id]
        observations = {value["invocation_id"]: value for value in task["observations"]}
        if entry.get("invocation_id") not in observations:
            raise ValueError(f"{item_id}: invocation is not approved in the frozen release")
        value = observations[entry["invocation_id"]]
        annotation = parse_signature(task["signature"]).return_annotation
        out.append(ResolvedItem(
            item_id=item_id, signature=task["signature"], docstring=task["specification"],
            inputs=decode_value(value["inputs"]), expected=decode_value(value["expected"]),
            invocation_id=value["invocation_id"], invocation_pool=value.get("pool"),
            input_line=value.get("input_line", ""), scoring_profile=task["scoring_profile"],
            return_type_source="signature" if annotation is not None else "none",
        ))
    return out


# HumanEval+ / MBPP+  (core)

def _resolve_evalplus(benchmark: str) -> Callable[[Dict[str, Any]], List[ResolvedItem]]:
    """Build a resolver using recorded IDs and specification-only return typing."""
    def resolve(manifest: Dict[str, Any]) -> List[ResolvedItem]:
        if manifest.get("data_policy") == "reviewed_release_v1":
            return _resolve_prepared(manifest, benchmark)
        from evalplus_data import LOADERS

        items = {}
        for item in LOADERS[benchmark]():
            if item.item_id in items:
                raise ValueError(f"{benchmark}: duplicate dataset item ID {item.item_id!r}")
            items[item.item_id] = item

        resolved: List[ResolvedItem] = []
        for entry in manifest_items(manifest, benchmark):
            item = items.get(entry["item_id"])
            if item is None or not item.eligible:
                raise KeyError(
                    f"{benchmark}: manifest item {entry['item_id']!r} is missing "
                    f"or ineligible in the current dataset; check the data and manifest."
                )
            invocations = {}
            for invocation in item.invocations:
                if invocation.invocation_id in invocations:
                    raise ValueError(
                        f"{benchmark}: item {item.item_id!r} has duplicate invocation "
                        f"ID {invocation.invocation_id!r}"
                    )
                invocations[invocation.invocation_id] = invocation
            wanted = entry.get("invocation_id")
            invocation = invocations.get(wanted)
            if invocation is None:
                raise KeyError(
                    f"{benchmark}: item {item.item_id!r} has no invocation "
                    f"{wanted!r}; check the data and manifest."
                )

            # Use the same signature parser as the runtime. Expected values
            # remain grading targets and supply no prediction-time type hint.
            annotation = parse_signature(item.signature).return_annotation
            resolved.append(ResolvedItem(
                item_id=item.item_id,
                signature=item.signature,
                docstring=item.docstring,
                inputs=invocation.inputs,
                expected=invocation.expected,
                invocation_id=invocation.invocation_id,
                input_line=invocation.input_line,
                invocation_pool=getattr(invocation, "pool", None),
                return_type_source="signature" if annotation is not None else "none",
            ))
        return resolved
    return resolve


def _compare_values(predicted: Any, expected: Any) -> bool:
    from scoring import results_match
    return results_match(predicted, expected)


# GSM8K  (generalization: numerical reasoning)

GSM8K_SIGNATURE = "solve_math(question: str) -> float"
GSM8K_DOCSTRING = ("Solve the given grade school math word problem. "
                   "Return ONLY the final numeric answer.")


def _resolve_gsm8k(manifest: Dict[str, Any]) -> List[ResolvedItem]:
    if manifest.get("data_policy") == "reviewed_release_v1":
        return _resolve_prepared(manifest, "gsm8k")
    from datasets import load_dataset
    from scoring import extract_gsm8k_answer
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out: List[ResolvedItem] = []
    for entry in manifest_items(manifest, "gsm8k"):
        item_id = entry["item_id"]
        if not isinstance(item_id, str) or re.fullmatch(r"gsm8k/(0|[1-9][0-9]*)", item_id) is None:
            raise ValueError(f"gsm8k: invalid item ID {item_id!r}")
        idx = int(item_id[len("gsm8k/"):])
        if idx >= len(ds):
            raise IndexError(f"gsm8k: item ID {item_id!r} is outside the dataset")
        row = ds[idx]
        expected = extract_gsm8k_answer(row["answer"])
        if not math.isfinite(expected):
            raise ValueError(f"gsm8k: item {item_id!r} has no finite numeric expected answer")
        out.append(ResolvedItem(
            item_id=item_id,
            signature=GSM8K_SIGNATURE,
            docstring=GSM8K_DOCSTRING,
            inputs={"question": row["question"]},
            expected=expected,
            return_type=float,
            input_line=row["question"][:120],
        ))
    return out


def _compare_gsm8k(predicted: Any, expected: Any) -> bool:
    from scoring import answers_match
    return answers_match(predicted, expected)


ADAPTERS: Dict[str, Adapter] = {
    "humaneval_plus": Adapter("humaneval_plus", _resolve_evalplus("humaneval_plus"), _compare_values),
    "mbpp_plus": Adapter("mbpp_plus", _resolve_evalplus("mbpp_plus"), _compare_values),
    "gsm8k": Adapter("gsm8k", _resolve_gsm8k, _compare_gsm8k),
}


def get_adapter(name: str) -> Adapter:
    try:
        return ADAPTERS[name]
    except KeyError:
        raise ValueError(f"unknown benchmark {name!r}. Known: {sorted(ADAPTERS)}")
