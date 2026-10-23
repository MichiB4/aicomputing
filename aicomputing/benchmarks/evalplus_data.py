"""
EvalPlus (HumanEval+ / MBPP+) adapters for spec-only execution.

These two benchmarks are the CORE specification-execution benchmarks. Both are
reduced here to the same shape the virtual runtime consumes:

    signature + docstring  (the specification)
    inputs                 (one concrete invocation)
    expected               (ground truth for that invocation)

One invocation per function
---------------------------
The primary experiment evaluates each function once. EvalPlus ships ~1000
inputs per HumanEval task, and evaluating all of them would make the "sample
size" an artefact of test-suite depth rather than of benchmark coverage: a
single function with 1100 stress inputs would outweigh a hundred others. It
would also break the paired analysis, whose unit is the function.

Which invocation is chosen matters. EvalPlus orders each suite with the
original hand-written cases first and the generated stress cases after, so
taking index 0 would systematically select the easiest input for every
function. selection.derive_index() gives each function an independent seeded
draw across its whole suite instead.

Where ground truth comes from
-----------------------------
No reference implementation is ever executed here.

  HumanEval+  the compiled `test` script assigns two literal lists inside
              check(): `inputs` and `results`, aligned pairwise. Both are read
              with ast.literal_eval. 158 of 164 tasks expose them this way;
              the other six build `results` from an expression and are
              reported as ineligible rather than silently dropped.

  MBPP+       a frozen ground-truth cache can supply base and plus invocations
              with expected values computed offline. Loading it does not
              execute the reference implementation. Otherwise, `test_list`
              supplies canonical positional
              `assert f(<literals>) == <literal>` cases; wrapped and
              keyword-bearing calls are skipped.

Invocation pool policy
----------------------
  HumanEval+  base + plus. The HF `inputs` list IS the expanded EvalPlus
              suite (HumanEval/0: 1006 = 7 base + 999 plus) and every entry
              has a precomputed expected value in the aligned `results` list.
              Verified pool: 1,492 base + 120,986 plus across 158 tasks.
              Each invocation is labelled base/plus BY CONTENT, because the
              HF build and the evalplus package release are different
              revisions -- positional labelling matched on only 65 of 158
              tasks, whereas content matching classified 100% with 0
              unlabelled.

  MBPP+       frozen invocations for cached items; available records can come
              from the base and plus pools. Items without cached ground truth
              use base-only cases parsed from `test_list`.
              `scripts/build_mbpp_ground_truth.py` executes canonical solutions
              offline to build that cache, separately from loading specifications.
              Each selected invocation records its actual `pool` label, and
              `invocation_pool_policy` summarizes those labels in the manifest.


What the model is shown
-----------------------
Only signature + docstring + the selected input. In particular:

  * the reference solution (`canonical_solution` / `code`) is used solely to
    recover MBPP's function SIGNATURE, which MBPP's natural-language prompt
    does not contain. The body is never shown to the model.
  * `>>>` doctest examples are stripped from HumanEval docstrings by default.
    They are part of the public specification rather than hidden tests, but
    they are worked input/output pairs, and leaving them in would make every
    arm partially few-shot on exactly the quantity being measured. Controlled
    by keep_doctests= so the decision is explicit and reversible.
"""

from __future__ import annotations

import ast
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aicomputing.types import parse_signature

# HuggingFace dataset IDs. Pinned here rather than inline so a provider change
# is a one-line edit; both are public and require no token.
HUMANEVALPLUS_DATASET = "evalplus/humanevalplus"
MBPPPLUS_DATASET = "evalplus/mbppplus"


@dataclass
class Invocation:
    """One concrete call of a benchmark function, with its ground truth."""
    invocation_id: str          # stable within the item, e.g. "in:37"
    inputs: Dict[str, Any]      # keyword form, as the runtime expects
    expected: Any
    input_line: str             # human-readable rendering, for the CSV
    pool: str = "unknown"       # "base" | "plus" | "unknown"


@dataclass
class SpecItem:
    """A function reduced to its specification plus available invocations."""
    item_id: str
    signature: str
    docstring: str
    entry_point: str
    invocations: List[Invocation] = field(default_factory=list)
    ineligible_reason: Optional[str] = None

    @property
    def eligible(self) -> bool:
        return self.ineligible_reason is None and bool(self.invocations)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _param_names(signature: str) -> List[str]:
    """
    Parameter names from a signature string, annotations and defaults removed.

    Parsed with ast rather than split(",") because annotations legitimately
    contain commas — `Dict[str, int]` would otherwise be read as two
    parameters and every input dict would be built with the wrong keys.
    """
    try:
        tree = ast.parse(f"def {signature.rstrip(':')}: pass")
    except SyntaxError:
        # Fall back to a bare-name signature with no annotations.
        inner = signature[signature.find("(") + 1: signature.rfind(")")]
        return [p.split("=")[0].split(":")[0].strip()
                for p in inner.split(",") if p.strip()]
    fn = tree.body[0]
    args = fn.args
    names = [a.arg for a in (*args.posonlyargs, *args.args)]
    if args.vararg:
        names.append(args.vararg.arg)
    names += [a.arg for a in args.kwonlyargs]
    return names


def _to_kwargs(values: Sequence[Any], names: Sequence[str]) -> Dict[str, Any]:
    """Zip positional test inputs onto parameter names."""
    out: Dict[str, Any] = {}
    for i, v in enumerate(values):
        out[names[i] if i < len(names) else f"arg{i}"] = v
    return out


def _render_call(entry_point: str, values: Sequence[Any]) -> str:
    return f"{entry_point}({', '.join(repr(v) for v in values)})"


# ---------------------------------------------------------------------------
# HumanEval+
# ---------------------------------------------------------------------------

def _humaneval_signature(prompt: str) -> str:
    """Last `def` line in the prompt, without the trailing colon."""
    sig = None
    for line in prompt.splitlines():
        s = line.strip()
        if s.startswith("def ") and s.endswith(":"):
            sig = s[len("def "):-1]
    if sig is None:
        raise ValueError("no function definition found in prompt")
    return sig


def _humaneval_docstring(prompt: str, *, keep_doctests: bool) -> str:
    """
    The task description from a HumanEval prompt.

    The prompt is a module: imports, the def line, then a docstring. Only the
    docstring is the specification; everything else is scaffolding the model
    does not need. `>>>` blocks are dropped unless keep_doctests is set (see
    the module docstring for why).
    """
    m = re.search(r'("""|\'\'\')(.*?)\1', prompt, re.S)
    body = m.group(2) if m else prompt
    lines = []
    for line in body.splitlines():
        if not keep_doctests and line.strip().startswith(">>>"):
            break
        lines.append(line)
    text = "\n".join(lines)
    # Collapse the indentation the prompt carries and tidy blank edges.
    text = "\n".join(l.strip() for l in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _humaneval_pairs(test_src: str) -> Tuple[List[Any], List[Any]]:
    """
    The `inputs` and `results` literal lists from an EvalPlus test script.

    Raises ValueError when either is absent or non-literal, which is how the
    six known ineligible tasks are detected.
    """
    tree = ast.parse(test_src)
    found: Dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    name = getattr(stmt.targets[0], "id", None)
                    if name in ("inputs", "results"):
                        try:
                            found[name] = ast.literal_eval(stmt.value)
                        except (ValueError, SyntaxError, TypeError):
                            raise ValueError(f"{name} is not a literal list")
    if "inputs" not in found or "results" not in found:
        raise ValueError("test script has no literal inputs/results pair")
    if len(found["inputs"]) != len(found["results"]):
        raise ValueError("inputs and results differ in length")
    if not found["inputs"]:
        raise ValueError("empty input list")
    return found["inputs"], found["results"]


def _humaneval_base_keys() -> Optional[Dict[str, set]]:
    """
    Per-task set of BASE inputs, used to label the HF pool base vs plus.

    The HuggingFace `evalplus/humanevalplus` build ships one flat `inputs`
    list with no base/plus marking, so the split has to come from the evalplus
    package, which exposes `base_input` and `plus_input` separately.

    Labelling is by CONTENT, not by index. Positional labelling was tried and
    is wrong: only 65 of 158 tasks have the HF list equal to
    base_input + plus_input in order, because the HF build and the package
    release are different revisions. Content matching classified 100% of the
    122,478-input pool when this was verified, with 0 unlabelled.

    Returns None when the package is unavailable; callers then record the pool
    as "unknown" rather than guessing.
    """
    try:
        from evalplus.data import get_human_eval_plus
    except Exception:
        return None
    try:
        tasks = get_human_eval_plus()
    except Exception:
        return None
    out: Dict[str, set] = {}
    for task_id, task in tasks.items():
        keys = set()
        for value in task.get("base_input", []) or []:
            try:
                keys.add(repr(value))
            except Exception:
                pass
        out[task_id] = keys
    return out


def load_humanevalplus(*, keep_doctests: bool = False) -> List[SpecItem]:
    """
    Load HumanEval+ as SpecItems. Ineligible tasks are returned, flagged.

    The selectable invocation pool is base + plus: the HF `inputs` list is the
    full EvalPlus-expanded suite (1006 for HumanEval/0 = 7 base + 999 plus),
    and each entry is labelled by content so the chosen invocation's origin is
    recorded. Ground truth comes from the aligned `results` list, so no
    reference implementation is executed.
    """
    from datasets import load_dataset

    base_keys = _humaneval_base_keys()
    items: List[SpecItem] = []
    for row in load_dataset(HUMANEVALPLUS_DATASET, split="test"):
        task_id = row["task_id"]
        try:
            signature = _humaneval_signature(row["prompt"])
            # Reject unusable signatures before they enter the selection pool.
            parse_signature(signature)
        except ValueError as exc:
            items.append(SpecItem(task_id, "", "", row["entry_point"],
                                  ineligible_reason=str(exc)))
            continue
        docstring = _humaneval_docstring(row["prompt"], keep_doctests=keep_doctests)
        names = _param_names(signature)
        try:
            raw_inputs, raw_results = _humaneval_pairs(row["test"])
        except (ValueError, SyntaxError) as exc:
            items.append(SpecItem(task_id, signature, docstring, row["entry_point"],
                                  ineligible_reason=f"no extractable ground truth: {exc}"))
            continue
        task_base = base_keys.get(task_id) if base_keys is not None else None
        invocations = []
        for i, (vals, exp) in enumerate(zip(raw_inputs, raw_results)):
            if task_base is None:
                pool = "unknown"
            else:
                try:
                    pool = "base" if repr(vals) in task_base else "plus"
                except Exception:
                    pool = "unknown"
            invocations.append(Invocation(
                invocation_id=f"in:{i}",
                inputs=_to_kwargs(vals, names),
                expected=exp,
                input_line=_render_call(row["entry_point"], vals),
                pool=pool,
            ))
        items.append(SpecItem(task_id, signature, docstring, row["entry_point"], invocations))
    return items


# ---------------------------------------------------------------------------
# MBPP+
# ---------------------------------------------------------------------------

def _mbpp_signature(code: str, entry_point: Optional[str] = None) -> Tuple[str, str]:
    """
    Signature and function name from MBPP's reference solution.

    Only the `def` line is read. MBPP's prompt is one English sentence with no
    interface, so the signature has to come from somewhere; the body is never
    used and never shown to the model.
    """
    # MBPP reference solutions contain regex literals written as plain strings
    # ("\w+"), so parsing them raises SyntaxWarning on 3.12+. The warning is
    # about the dataset's code, not ours, and there is nothing to fix in it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(code)
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not defs:
        raise ValueError("reference solution defines no function")
    fn = next((d for d in defs if d.name == entry_point), defs[-1]) if entry_point else defs[-1]
    args = ", ".join(a.arg for a in (*fn.args.posonlyargs, *fn.args.args))
    return f"{fn.name}({args})", fn.name


def _mbpp_invocations(test_list: Sequence[str], entry_point: str,
                      names: Sequence[str]) -> List[Invocation]:
    """Parse canonical positional `assert f(<literals>) == <literal>` asserts."""
    out: List[Invocation] = []
    for i, src in enumerate(test_list):
        try:
            stmt = ast.parse(src.strip()).body[0]
        except (SyntaxError, IndexError):
            continue
        if not isinstance(stmt, ast.Assert):
            continue
        cmp_ = stmt.test
        if not (isinstance(cmp_, ast.Compare) and len(cmp_.ops) == 1
                and isinstance(cmp_.ops[0], ast.Eq)):
            continue
        call = cmp_.left
        # A wrapped call (set(f(x)) == ...) means the comparison itself carries
        # semantics; the operand is not the function's return value.
        if not (isinstance(call, ast.Call) and getattr(call.func, "id", None) == entry_point):
            continue
        # Keyword arguments cannot be reconstructed by positional mapping.
        if call.keywords:
            continue
        try:
            values = [ast.literal_eval(a) for a in call.args]
            expected = ast.literal_eval(cmp_.comparators[0])
        except (ValueError, SyntaxError, TypeError):
            continue
        out.append(Invocation(
            invocation_id=f"assert:{i}",
            inputs=_to_kwargs(values, names),
            expected=expected,
            input_line=_render_call(entry_point, values),
            pool="base",
        ))
    return out


MBPP_GROUND_TRUTH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "mbpp_plus_ground_truth.json")


def load_mbpp_ground_truth(path: str = MBPP_GROUND_TRUTH) -> Optional[Dict[str, Any]]:
    """
    Load the frozen MBPP+ expected-output cache, if it has been built.

    Built offline by scripts/build_mbpp_ground_truth.py, which executes the
    official canonical solutions ONCE to compute expected outputs and freezes
    them. The artifact contains only item IDs, inputs, expected values and
    provenance -- never the implementation -- so nothing downstream can expose
    a reference solution even by accident.

    Decode typed inputs and expected values. Legacy or unknown cache formats
    must be rebuilt; their original value types cannot be inferred safely.
    Returns None when the cache is absent, in which case MBPP+ falls back to
    base-only invocations from `test_list`.
    """
    if not os.path.exists(path):
        return None
    from mbpp_ground_truth import (
        GROUND_TRUTH_VERSION, VALUE_ENCODING, decode_value,
    )

    with open(path, encoding="utf-8") as fh:
        cache = json.load(fh)
    if (type(cache) is not dict
            or type(cache.get("ground_truth_version")) is not int
            or cache["ground_truth_version"] != GROUND_TRUTH_VERSION
            or cache.get("value_encoding") != VALUE_ENCODING):
        raise ValueError(
            "Unsupported MBPP ground-truth cache; rebuild it with "
            "scripts/build_mbpp_ground_truth.py"
        )
    for task in cache["tasks"].values():
        for inv in task["invocations"]:
            inv["inputs"] = decode_value(inv["inputs"])
            inv["expected"] = decode_value(inv["expected"])
    return cache


def load_mbppplus(*, ground_truth_path: str = MBPP_GROUND_TRUTH) -> List[SpecItem]:
    """
    Load MBPP+ as SpecItems. Ineligible items are returned, flagged.

    Cached items use their frozen invocation records. Other items fall back
    to base-only cases from `test_list`. Each invocation's `pool` label records
    the actual source used.
    """
    from datasets import load_dataset

    cache = load_mbpp_ground_truth(ground_truth_path)
    cached_tasks = (cache or {}).get("tasks", {})
    items: List[SpecItem] = []
    for row in load_dataset(MBPPPLUS_DATASET, split="test"):
        item_id = f"Mbpp/{row['task_id']}"
        try:
            signature, entry_point = _mbpp_signature(row["code"])
        except (ValueError, SyntaxError) as exc:
            items.append(SpecItem(item_id, "", row["prompt"], "",
                                  ineligible_reason=f"unparseable reference: {exc}"))
            continue
        names = _param_names(signature)
        frozen = cached_tasks.get(item_id)
        if frozen:
            # base + plus, from the frozen cache.
            invocations = [
                Invocation(
                    invocation_id=f"gt:{inv['index']}",
                    inputs=_to_kwargs(inv["inputs"], names),
                    expected=inv["expected"],
                    input_line=_render_call(entry_point, inv["inputs"]),
                    pool=inv["pool"],
                )
                for inv in frozen["invocations"]
            ]
        else:
            # base only, parsed from the published asserts.
            invocations = _mbpp_invocations(row["test_list"] or [], entry_point, names)
        reason = None if invocations else "no canonical `assert f(...) == literal` test"
        items.append(SpecItem(item_id, signature, (row["prompt"] or "").strip(),
                              entry_point, invocations, ineligible_reason=reason))
    return items


LOADERS = {
    "humaneval_plus": load_humanevalplus,
    "mbpp_plus": load_mbppplus,
}
