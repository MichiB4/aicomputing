"""
Benchmark adapters: manifest entry -> executable specification + ground truth.

Each adapter turns the IDs recorded in the experiment manifest back into the
three things the virtual runtime needs (signature, docstring, inputs) plus the
expected value, and supplies the comparison used to score the answer.

Scoring functions are imported from the legacy oracle modules rather than
reimplemented. They are the same graders the historical results were produced
with, so a number computed here is comparable to one computed before the
refactor, and there is only one place to fix a grading bug.

The adapters own the specification text. That is deliberate: the docstring IS
the program under this paradigm, so a change to it is a change to the
experiment, and having them all in one file makes that reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from experiment import BENCHMARK_GROUP, manifest_items


@dataclass
class ResolvedItem:
    """One manifest entry, resolved into something executable."""
    item_id: str
    signature: str
    docstring: str
    inputs: Dict[str, Any]
    expected: Any
    return_type: Optional[type] = None
    category: Optional[str] = None
    invocation_id: Optional[str] = None
    input_line: str = ""
    invocation_pool: Optional[str] = None    # "base" | "plus" | "unknown"
    return_type_source: str = "signature"    # "signature" | "expected" | "none"


@dataclass
class Adapter:
    name: str
    resolve: Callable[[Dict[str, Any]], List[ResolvedItem]]
    compare: Callable[[Any, Any], bool]

    @property
    def group(self) -> str:
        return BENCHMARK_GROUP[self.name]


# ---------------------------------------------------------------------------
# HumanEval+ / MBPP+  (core)
# ---------------------------------------------------------------------------

def _resolve_evalplus(benchmark: str):
    def resolve(manifest: Dict[str, Any]) -> List[ResolvedItem]:
        from evalplus_data import LOADERS
        items = {i.item_id: i for i in LOADERS[benchmark]()}
        out: List[ResolvedItem] = []
        for entry in manifest_items(manifest, benchmark):
            item = items.get(entry["item_id"])
            if item is None or not item.eligible:
                # The manifest names an item this dataset revision no longer
                # provides. Skipping silently would shrink the sample without
                # trace, so it is surfaced as a hard error instead.
                raise KeyError(
                    f"{benchmark}: manifest item {entry['item_id']!r} is missing "
                    f"or ineligible in the current dataset. The manifest and the "
                    f"data have diverged; rebuild the manifest."
                )
            wanted = entry.get("invocation_id")
            inv = next((v for v in item.invocations if v.invocation_id == wanted), None)
            if inv is None:
                raise KeyError(
                    f"{benchmark}: item {item.item_id} has no invocation "
                    f"{wanted!r}; the test suite changed under the manifest."
                )
            rt, rt_source = _evaluator_return_type(item.signature, inv.expected)
            out.append(ResolvedItem(
                item_id=item.item_id,
                signature=item.signature,
                docstring=item.docstring,
                inputs=inv.inputs,
                expected=inv.expected,
                return_type=rt,
                invocation_id=inv.invocation_id,
                input_line=inv.input_line,
                invocation_pool=getattr(inv, "pool", None),
                return_type_source=rt_source,
            ))
        return out
    return resolve


def _evaluator_return_type(signature: str, expected: Any):
    """
    Structural type for output validation when the signature declares none.

    MBPP signatures are recovered from the reference solution's `def` line and
    carry no return annotation, so STRICT validation would have nothing to
    check and would silently degrade to "anything goes" while still being
    reported as strict. The evaluator therefore takes the type - never the
    value - of the known expected result.

    The distinction that makes this sound:

        expected-result TYPE is used by the evaluator, after the model has
        answered, to check the shape of what came back;
        expected-result CONTENT is never placed in a prompt.

    The model is not told the return type, and nothing here influences
    generation - ExecuteRequest.return_type feeds parsing and validation only,
    while prompts are built from signature + docstring + inputs. Returns
    (type, source); source "none" means no structural guarantee was available
    and the row records that rather than claiming strict typing.
    """
    if "->" in signature:
        return None, "signature"          # the declared annotation governs
    if expected is None:
        return None, "none"               # NoneType would be meaningless
    t = type(expected)
    if t in (int, float, str, bool, list, dict, tuple, set):
        return t, "expected"
    return None, "none"


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
        idx = int(entry["item_id"].split("/")[-1])
        row = ds[idx]
        out.append(ResolvedItem(
            item_id=entry["item_id"],
            signature=GSM8K_SIGNATURE,
            docstring=GSM8K_DOCSTRING,
            inputs={"question": row["question"]},
            expected=extract_gsm8k_answer(row["answer"]),
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
