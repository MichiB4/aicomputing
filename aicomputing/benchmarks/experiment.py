"""
Experiment configuration, item pools, and the reproducibility manifest.

This module is the single place that decides *which* examples the experiment
runs on. Oracles read the manifest and evaluate exactly what it names; they no
longer sample anything themselves. That is what makes the paired analysis
possible: every model and every strategy is handed an identical item set, so
`benchmark_item_id` joins across cells.

Benchmark groups
----------------
The four benchmarks are not interchangeable and must not be pooled into a
single "AI computing accuracy":

    core            humaneval_plus, mbpp_plus
                    Specification-execution proper: a Python function contract
                    with a concrete invocation.

    generalization  gsm8k, bbh
                    Whether the same mechanism carries to numerical (gsm8k)
                    and symbolic/logical (bbh) reasoning expressed as a
                    function contract.

Reporting is per benchmark, with `benchmark_group` available for grouping.
A mean over all four would weight an arbitrary sampling decision (200 GSM8K
problems) against a fixed benchmark size (158 HumanEval+ functions) and would
change meaning if either sample size were adjusted.

Item identity
-------------
HumanEval+/MBPP+ carry native task IDs. GSM8K and BBH do not, so their IDs are
positional within a dataset split (`gsm8k/117`, `web_of_lies/42`). Positional
IDs are only as stable as the dataset revision, so the manifest records a
fingerprint of each split; check_pools() reports a mismatch rather than
silently evaluating different problems under the same ID.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from selection import (
    SELECTION_ALGORITHM_VERSION,
    derive_index,
    select_items,
    select_stratified,
)

# Bumped when the manifest's own structure changes incompatibly.
MANIFEST_VERSION = 1

# Project-wide seed. Nothing else in the repository fixes a seed, so this
# establishes the convention rather than competing with one.
DEFAULT_SEED = 42

CORE_BENCHMARKS = ("humaneval_plus", "mbpp_plus")
GENERALIZATION_BENCHMARKS = ("gsm8k", "bbh")
ALL_BENCHMARKS = CORE_BENCHMARKS + GENERALIZATION_BENCHMARKS

BENCHMARK_GROUP: Dict[str, str] = {
    **{b: "core" for b in CORE_BENCHMARKS},
    **{b: "generalization" for b in GENERALIZATION_BENCHMARKS},
}

# The code-generation baseline answers "would generating an implementation have
# done better?", which is only a meaningful question where the benchmark item
# IS a function to implement. On GSM8K and BBH the model would have to invent a
# program for a one-off word problem, so a failure there measures the framing,
# not code generation. Restricted rather than merely discouraged: the runner
# refuses the combination.
BASELINE_BENCHMARKS = frozenset(CORE_BENCHMARKS)
BASELINE_NAME = "code_gen_baseline"

# BBH categories carried into the experiment. Each needs a hand-written
# signature and docstring in bbh_oracle.TASK_SPECS, so this list is a
# methodological choice, not a lookup - adding a category means authoring a
# specification for it.
#
# This is a FIVE-TASK SUBSET, not BIG-Bench Hard. BBH has 27 tasks; results
# here cover five of them and must be reported as BBH-5. Calling it "BBH"
# would claim coverage the experiment does not have. BBH_DISPLAY_NAME is what
# goes in metadata and tables; the original task names are preserved
# unchanged in every result row via benchmark_category.
BBH_DISPLAY_NAME = "BBH-5"
BBH_SUBSET_NOTE = ("five-task subset of BIG-Bench Hard; not the full "
                   "27-task suite")

BBH_TASKS = (
    "date_understanding",
    "boolean_expressions",
    "tracking_shuffled_objects_three_objects",
    "logical_deduction_five_objects",
    "web_of_lies",
)

# Primary experiment sample sizes (requirement 14). HumanEval+ takes every
# eligible function; the others are sampled.
MAIN_SIZES: Dict[str, Optional[int]] = {
    "humaneval_plus": None,   # None = all eligible
    "mbpp_plus": 200,
    "gsm8k": 200,
    "bbh": 200,
}

# Pilot-v2 smoke sizes. Purpose is plumbing verification, not results.
PILOT_SIZES: Dict[str, Optional[int]] = {
    "humaneval_plus": 5,
    "mbpp_plus": 5,
    "gsm8k": 10,
    "bbh": 10,          # stratified -> 2 per task across all five
}


# ---------------------------------------------------------------------------
# Item pools
# ---------------------------------------------------------------------------

@dataclass
class PoolItem:
    """One candidate benchmark item, before selection."""
    item_id: str
    signature: str
    docstring: str
    category: Optional[str] = None          # BBH task; None elsewhere
    invocation_ids: List[str] = field(default_factory=list)
    payload: Any = None                     # loader-specific extra data


def content_fingerprint(*, signature: str, docstring: str, inputs: Any,
                        expected: Any, category: Optional[str] = None) -> str:
    """
    Digest of an item's SEMANTIC content, independent of its position.

    GSM8K and BBH have no native IDs, so theirs are positional
    (`gsm8k/117`). A positional ID is only meaningful relative to a fixed
    dataset ordering: if a revision inserts or reorders rows, `gsm8k/117`
    silently becomes a different problem and the manifest keeps "reproducing"
    an experiment that is no longer the same one. The earlier claim that
    selection survived reordering held for the SELECTION step but not for
    resolution, which is the gap this closes.

    The fingerprint covers everything that makes the observation what it is:
    the specification the model sees, the concrete inputs, and the expected
    answer. Two items with the same fingerprint are the same observation
    wherever they sit in the file.
    """
    h = hashlib.blake2b(digest_size=16)
    for part in (signature, docstring, category or "",
                 canonical_repr(inputs), canonical_repr(expected)):
        h.update(str(part).encode("utf-8"))
        h.update(b"\x1f")   # unit separator: cannot occur in the content
    return h.hexdigest()


def canonical_repr(value: Any) -> str:
    """
    Deterministic serialisation of a structured Python value.

    repr() is not a stable serialisation for this purpose. Dict ordering
    follows insertion, so two loaders building the same mapping in a different
    order produce different reprs; set ordering depends on hash randomisation
    and varies between processes for strings. Either would make a fingerprint
    disagree with itself across runs and report drift where there is none.

    Types are kept distinct so that 1, 1.0 and True do not collapse onto one
    another -- an expected value of True and one of 1 are different ground
    truths even though Python compares them equal.
    """
    if isinstance(value, bool):
        return f"b:{value}"
    if value is None:
        return "n:"
    if isinstance(value, int):
        return f"i:{value}"
    if isinstance(value, float):
        # repr() round-trips floats exactly in Python 3; format() would not.
        return f"f:{value!r}"
    if isinstance(value, str):
        return f"s:{value}"
    if isinstance(value, bytes):
        return f"y:{value.hex()}"
    if isinstance(value, (list, tuple)):
        tag = "l" if isinstance(value, list) else "t"
        return tag + "[" + ",".join(canonical_repr(v) for v in value) + "]"
    if isinstance(value, (set, frozenset)):
        tag = "S" if isinstance(value, set) else "F"
        # Sorted by canonical form: a total order independent of hash
        # randomisation, unlike sorting the elements themselves.
        return tag + "{" + ",".join(sorted(canonical_repr(v) for v in value)) + "}"
    if isinstance(value, dict):
        items = sorted((canonical_repr(k), canonical_repr(v))
                       for k, v in value.items())
        return "d{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    # Anything else: tagged repr, so it cannot collide with a canonical form.
    return f"r:{value!r}"


def _fingerprint(values: Sequence[str]) -> str:
    """Short stable digest of a pool, to detect a dataset revision change."""
    h = hashlib.blake2b(digest_size=8)
    for v in values:
        h.update(str(v).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def dataset_revision(benchmark: str) -> Optional[str]:
    """
    Source revision for a benchmark's data, where the loader exposes one.

    `datasets` computes a content fingerprint per split, which changes if the
    underlying data changes; that is recorded alongside the version string.
    Returns None rather than a guess when nothing is available.
    """
    try:
        if benchmark == "gsm8k":
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split="test")
        elif benchmark == "bbh":
            from datasets import load_dataset
            ds = load_dataset("lukaemon/bbh", BBH_TASKS[0], split="test")
        elif benchmark == "humaneval_plus":
            from datasets import load_dataset
            ds = load_dataset("evalplus/humanevalplus", split="test")
        elif benchmark == "mbpp_plus":
            from datasets import load_dataset
            ds = load_dataset("evalplus/mbppplus", split="test")
        else:
            return None
    except Exception:
        return None
    version = getattr(getattr(ds, "info", None), "version", None)
    fp = getattr(ds, "_fingerprint", None)
    parts = [str(x) for x in (version, fp) if x]
    return "/".join(parts) or None


def load_pool(benchmark: str) -> List[PoolItem]:
    """Load every candidate item for a benchmark, before selection."""
    if benchmark in ("humaneval_plus", "mbpp_plus"):
        from evalplus_data import LOADERS
        items = LOADERS[benchmark]()
        return [
            PoolItem(
                item_id=i.item_id,
                signature=i.signature,
                docstring=i.docstring,
                invocation_ids=[v.invocation_id for v in i.invocations],
                payload=i,
            )
            for i in items if i.eligible
        ]

    if benchmark == "gsm8k":
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [
            PoolItem(item_id=f"gsm8k/{i}", signature="", docstring="", payload=row)
            for i, row in enumerate(ds)
        ]

    if benchmark == "bbh":
        from datasets import load_dataset
        out: List[PoolItem] = []
        for task in BBH_TASKS:
            ds = load_dataset("lukaemon/bbh", task, split="test")
            for i, row in enumerate(ds):
                out.append(PoolItem(item_id=f"{task}/{i}", signature="",
                                    docstring="", category=task, payload=row))
        return out

    raise ValueError(f"unknown benchmark {benchmark!r}. Known: {ALL_BENCHMARKS}")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_manifest(
    *,
    experiment_id: str,
    models: Sequence[str],
    strategies: Sequence[str],
    sizes: Dict[str, Optional[int]],
    benchmarks: Sequence[str] = ALL_BENCHMARKS,
    seed: int = DEFAULT_SEED,
    validation_mode: str = "strict",
    include_baseline: bool = True,
    generation: Optional[Dict[str, Any]] = None,
    model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    attach_fingerprints: bool = True,
) -> Dict[str, Any]:
    """
    Perform every selection and return the manifest as a plain dict.

    Selection happens once, here. Reruns load the result rather than
    re-deriving it, so a change to a dataset revision surfaces as a
    fingerprint mismatch instead of a quietly different sample.
    """
    manifest: Dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "selection_algorithm_version": SELECTION_ALGORITHM_VERSION,
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "validation_mode": validation_mode,
        "models": list(models),
        "strategies": list(strategies),
        "baseline": {
            "name": BASELINE_NAME,
            "benchmarks": sorted(BASELINE_BENCHMARKS & set(benchmarks)),
        } if include_baseline else None,
        # Everything that defines the experimental condition lives here, so
        # that code version + manifest determines the configuration.
        #
        # max_new_tokens used to be an unrecorded external fact: the runner
        # carried a MODEL_MAX_TOKENS table, so changing a per-model cap changed
        # the experiment without changing the manifest, and a row could not be
        # audited for the budget it actually ran under.
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_retries": 1,
            "max_new_tokens": None,      # None = the strategy's own budget
            **(generation or {}),
        },
        # Per-model overrides, resolved against `generation` at run time. Empty
        # until real model IDs are chosen: provider catalogs drift, and an
        # unverified ID recorded here would be worse than none.
        "model_configs": model_configs or {},
        # What each strategy asks the backend for. Recorded so the condition is
        # auditable from the manifest rather than inferred from the code, and
        # so a later change to a strategy default is visible as a manifest diff.
        "strategy_thinking_policy": {
            "direct": {"think": False},
            "cot": {"think": True},
            "pseudocode": {"think": True},
            "confidence": {"think": "model-decided"},
            "contract": {"predict": False, "check": True},
        },
        "benchmarks": {},
    }

    for bench in benchmarks:
        pool = load_pool(bench)
        by_id = {p.item_id: p for p in pool}
        k = sizes.get(bench)

        if bench == "bbh":
            # Stratified: never a pooled prefix (requirement 4).
            strata = {t: [p.item_id for p in pool if p.category == t] for t in BBH_TASKS}
            chosen = select_stratified(strata, k or len(pool), seed, salt=bench)
            ids = [i for t in sorted(chosen) for i in chosen[t]]
            stratum_counts = {t: len(v) for t, v in chosen.items()}
        else:
            all_ids = [p.item_id for p in pool]
            ids = select_items(all_ids, k if k is not None else len(all_ids),
                               seed, salt=bench)
            stratum_counts = None

        entries = []
        for item_id in ids:
            p = by_id[item_id]
            entry: Dict[str, Any] = {"item_id": item_id}
            if p.category:
                entry["category"] = p.category
            if p.invocation_ids:
                # One invocation per function (requirement 2/14), drawn per
                # item so the choice is independent of the sample size.
                idx = derive_index(seed, item_id, len(p.invocation_ids))
                entry["invocation_id"] = p.invocation_ids[idx]
            entries.append(entry)

        manifest["benchmarks"][bench] = {
            "group": BENCHMARK_GROUP[bench],
            "display_name": BBH_DISPLAY_NAME if bench == "bbh" else bench,
            "subset_note": BBH_SUBSET_NOTE if bench == "bbh" else None,
            "requested": k,
            "selected": len(entries),
            "pool_size": len(pool),
            "pool_fingerprint": _fingerprint(sorted(by_id)),
            "dataset_revision": dataset_revision(bench),
            # Filled in by _attach_content_fingerprints() from the pools the
            # SELECTED invocations actually came from. It was hard-coded here
            # and went stale the moment MBPP+ gained a frozen plus ground
            # truth: the manifest claimed base_only while selecting plus
            # invocations, so its own provenance contradicted its contents.
            "invocation_pool_policy": None,
            "strata": stratum_counts,
            "items": entries,
        }

    if attach_fingerprints:
        _attach_content_fingerprints(manifest)

    return manifest


def _attach_content_fingerprints(manifest: Dict[str, Any]) -> None:
    """
    Resolve every selected item and store a digest of its semantic content.

    Done as a second pass through the same adapters the run uses, so the
    fingerprint is taken from exactly the specification/inputs/expected triple
    that will later be executed - not from a parallel reimplementation that
    could drift from it.
    """
    from adapters import get_adapter          # late: adapters imports this module

    for bench, entry in manifest["benchmarks"].items():
        try:
            resolved = {i.item_id: i for i in get_adapter(bench).resolve(manifest)}
        except Exception as exc:              # data unavailable at build time
            entry["content_fingerprints"] = None
            entry["content_fingerprint_error"] = f"{type(exc).__name__}: {exc}"
            continue
        for item in entry["items"]:
            r = resolved.get(item["item_id"])
            if r is None:
                continue
            item["content_fingerprint"] = content_fingerprint(
                signature=r.signature, docstring=r.docstring,
                inputs=r.inputs, expected=r.expected, category=r.category,
            )
            if getattr(r, "invocation_pool", None):
                item["invocation_pool"] = r.invocation_pool
        entry["content_fingerprints"] = True
        entry["invocation_pool_policy"] = _describe_pool_policy(bench, entry["items"])


def _describe_pool_policy(benchmark: str, items) -> Optional[str]:
    """
    Describe where a benchmark's selected invocations actually came from.

    Derived from the selection rather than asserted, so the manifest cannot
    describe a source it is not using. The previous hard-coded string survived
    the arrival of the frozen MBPP+ ground truth and left every manifest
    claiming base_only while selecting plus invocations.
    """
    pools = {i.get("invocation_pool") for i in items if i.get("invocation_pool")}
    if not pools:
        return None
    if benchmark == "mbpp_plus":
        if "plus" in pools:
            return ("base+plus (plus expected outputs precomputed offline "
                    "from canonical solutions; see "
                    "scripts/build_mbpp_ground_truth.py)")
        return "base_only (frozen plus ground truth unavailable)"
    if pools == {"base"}:
        return "base_only"
    return "base+plus"


def verify_content(manifest: Dict[str, Any], benchmark: str, resolved,
                   *, hard: bool = True) -> List[str]:
    """
    Check that the items now being loaded are the ones the manifest selected.

    This is the check that makes positional IDs safe. `gsm8k/117` means
    nothing on its own; the fingerprint says which problem it was when the
    manifest was built, so a reordered or revised dataset is detected instead
    of silently substituting different observations.

    hard=True (the main experiment) raises on any mismatch: a reproducibility
    claim that quietly evaluates different data is worse than a failed run.
    """
    stored = {i["item_id"]: i.get("content_fingerprint")
              for i in manifest_items(manifest, benchmark)}
    problems: List[str] = []
    for item in resolved:
        want = stored.get(item.item_id)
        if not want:
            continue
        got = content_fingerprint(
            signature=item.signature, docstring=item.docstring,
            inputs=item.inputs, expected=item.expected, category=item.category,
        )
        if got != want:
            problems.append(
                f"{benchmark}:{item.item_id} content changed "
                f"({want[:12]} -> {got[:12]})"
            )
    if problems and hard:
        raise RuntimeError(
            f"{len(problems)} manifest item(s) no longer match the data they "
            f"were selected from. The dataset has changed underneath the "
            f"manifest, so this run would not reproduce the recorded "
            f"experiment. First: {problems[0]}"
        )
    return problems


def save_manifest(manifest: Dict[str, Any], path: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return path


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    check_compatible(manifest)
    return manifest


def check_compatible(manifest: Dict[str, Any]) -> None:
    """Refuse a manifest this code cannot reproduce faithfully."""
    mv = manifest.get("manifest_version")
    if mv != MANIFEST_VERSION:
        raise ValueError(
            f"manifest_version {mv} != {MANIFEST_VERSION} supported by this code. "
            f"Rebuild the manifest, or check out the revision that wrote it."
        )
    sv = manifest.get("selection_algorithm_version")
    if sv != SELECTION_ALGORITHM_VERSION:
        raise ValueError(
            f"selection_algorithm_version {sv} != {SELECTION_ALGORITHM_VERSION}. "
            f"The same seed would now select different items; rebuild the manifest."
        )


def manifest_items(manifest: Dict[str, Any], benchmark: str) -> List[Dict[str, Any]]:
    entry = manifest["benchmarks"].get(benchmark)
    if entry is None:
        raise ValueError(
            f"benchmark {benchmark!r} is not in this manifest "
            f"(has: {sorted(manifest['benchmarks'])})"
        )
    return entry["items"]


def check_pools(manifest: Dict[str, Any], benchmark: str) -> Optional[str]:
    """
    Compare the current pool against the fingerprint recorded at build time.

    Returns None when they agree, otherwise a message. A mismatch means the
    dataset changed underneath the manifest — for GSM8K and BBH, whose IDs are
    positional, that would silently repoint every ID at a different problem.
    """
    entry = manifest["benchmarks"].get(benchmark)
    if not entry:
        return None
    current = _fingerprint(sorted(p.item_id for p in load_pool(benchmark)))
    recorded = entry.get("pool_fingerprint")
    if recorded and current != recorded:
        return (f"{benchmark}: pool fingerprint changed "
                f"({recorded} -> {current}); item IDs may no longer refer to "
                f"the same examples")
    return None
