"""
Benchmark adapters: manifest entry -> executable specification + ground truth.

Each adapter turns the IDs recorded in the experiment manifest back into the
three things the virtual runtime needs (signature, docstring, inputs) plus the
expected value, and supplies the comparison used to score the answer.

Comparison functions reuse the legacy benchmark graders. End-to-end scoring
also depends on parsing, validation and failure handling in the runtime and
runner; sharing a comparison function alone does not establish historical
result comparability.

Generalization specification text is declared here. Core benchmark signatures
and docstrings come from the EvalPlus loaders. Selected specifications, inputs
and expected outputs are covered by the manifest's content fingerprints.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from aicomputing.types import parse_signature

from experiment import BENCHMARK_GROUP, manifest_items


@dataclass
class ResolvedItem:
    """One selected observation with execution inputs and grading metadata.

    The runner sends signature, docstring, inputs and an optional return-type
    override to the runtime. The expected value is retained for grading.
    Remaining fields identify the observation and its recorded sources.
    """
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


@dataclass
class Adapter:
    """Resolve a benchmark selection and compare predictions with expected values."""
    name: str
    resolve: Callable[[Dict[str, Any]], List[ResolvedItem]]
    compare: Callable[[Any, Any], bool]

    @property
    def group(self) -> str:
        return BENCHMARK_GROUP[self.name]


# ---------------------------------------------------------------------------
# HumanEval+ / MBPP+  (core)
# ---------------------------------------------------------------------------

def _resolve_evalplus(benchmark: str) -> Callable[[Dict[str, Any]], List[ResolvedItem]]:
    """Build a resolver using recorded IDs and specification-only return typing."""
    def resolve(manifest: Dict[str, Any]) -> List[ResolvedItem]:
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
    from humaneval_oracle import results_match
    return results_match(predicted, expected)


# ---------------------------------------------------------------------------
# GSM8K  (generalization: numerical reasoning)
# ---------------------------------------------------------------------------

GSM8K_SIGNATURE = "solve_math(question: str) -> float"
GSM8K_DOCSTRING = ("Solve the given grade school math word problem. "
                   "Return ONLY the final numeric answer.")


def _resolve_gsm8k(manifest: Dict[str, Any]) -> List[ResolvedItem]:
    from datasets import load_dataset
    from gsm8k_oracle import extract_gsm8k_answer
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
    from gsm8k_oracle import answers_match
    return answers_match(predicted, expected)


# ---------------------------------------------------------------------------
# BBH  (generalization: symbolic / logical reasoning)
# ---------------------------------------------------------------------------

def _resolve_bbh(manifest: Dict[str, Any]) -> List[ResolvedItem]:
    from datasets import load_dataset
    from bbh_oracle import TASK_SPECS
    cache: Dict[str, Any] = {}
    out: List[ResolvedItem] = []
    for entry in manifest_items(manifest, "bbh"):
        task, idx = entry["item_id"].rsplit("/", 1)
        if task not in cache:
            cache[task] = load_dataset("lukaemon/bbh", task, split="test")
        row = cache[task][int(idx)]
        spec = TASK_SPECS[task]
        # The parameter name is read from the declared signature rather than
        # hardcoded to "question". Passing {"question": ...} against a
        # signature declaring `expression` made the specification internally
        # inconsistent: harmless for the spec-only arms, which only render the
        # inputs into a prompt, but a hard TypeError for the code-generation
        # baseline, which actually calls the function.
        param = spec["signature"].split("(", 1)[1].split(":", 1)[0].strip()
        out.append(ResolvedItem(
            item_id=entry["item_id"],
            signature=spec["signature"],
            docstring=spec["docstring"],
            inputs={param: row["input"]},
            expected=row["target"],
            return_type=str,
            category=entry.get("category", task),
            input_line=row["input"][:120],
        ))
    return out


def _compare_bbh(predicted: Any, expected: Any) -> bool:
    from bbh_oracle import normalize_answer
    pred = "" if predicted is None else str(predicted)
    return normalize_answer(pred) == normalize_answer(str(expected))


ADAPTERS: Dict[str, Adapter] = {
    "humaneval_plus": Adapter("humaneval_plus", _resolve_evalplus("humaneval_plus"), _compare_values),
    "mbpp_plus": Adapter("mbpp_plus", _resolve_evalplus("mbpp_plus"), _compare_values),
    "gsm8k": Adapter("gsm8k", _resolve_gsm8k, _compare_gsm8k),
    "bbh": Adapter("bbh", _resolve_bbh, _compare_bbh),
}


def get_adapter(name: str) -> Adapter:
    try:
        return ADAPTERS[name]
    except KeyError:
        raise ValueError(f"unknown benchmark {name!r}. Known: {sorted(ADAPTERS)}")
