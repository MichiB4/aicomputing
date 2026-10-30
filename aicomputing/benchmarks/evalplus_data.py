"""Load HumanEval+/MBPP+ specifications and invocations without running solutions.

The manifest selects one seeded invocation per function. HumanEval targets come
from aligned literal inputs/results lists; nonliteral targets are ineligible.
Base/plus labels use content matching because dataset/package ordering differs.

MBPP prefers published literal input/output lists. Public base asserts cover
sources without precomputed output lists.

Models receive only signature, docstring, and selected inputs. Reference code
supplies MBPP signatures only. HumanEval doctests are stripped unless requested;
the reviewed-release workflow defines its own public-example policy.
HumanEval/38 adds an AST-verified public-helper description, never executable
helper code. Explicit specification exclusions record their reasons below."""

from __future__ import annotations

import ast
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aicomputing.types import parse_signature

# Public HuggingFace dataset IDs.
HUMANEVALPLUS_DATASET = "evalplus/humanevalplus"
MBPPPLUS_DATASET = "evalplus/mbppplus"

_MBPP_SPECIFICATION_EXCLUSIONS = {
    "Mbpp/419": (
        "specification asks to print a value, but the frozen reference grades "
        "a returned number; print-versus-return behavior is inconsistent"
    ),
}


@dataclass
class Invocation:
    """One concrete call of a benchmark function, with its ground truth."""
    invocation_id: str          # stable within the item, e.g. "in:37"
    inputs: Dict[str, Any]      # keyword form, as the runtime expects
    expected: Any
    input_line: str             # Human-readable call.
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


# Shared helpers

def _param_names(signature: str) -> List[str]:
    """Extract parameter names with AST parsing so generic-annotation commas are safe.

    Malformed headers fall back to bare-name parsing."""
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


# HumanEval+

def _humaneval_target(prompt: str, entry_point: Optional[str] = None) -> Tuple[ast.Module, ast.FunctionDef]:
    """Locate one declared top-level function without evaluating its module."""
    module = ast.parse(prompt)
    functions = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    targets = [node for node in functions if node.name == entry_point] if entry_point is not None else functions
    if len(targets) != 1:
        raise ValueError("HumanEval requires one unambiguous declared entry point")
    if isinstance(targets[0], ast.AsyncFunctionDef):
        raise ValueError("asynchronous HumanEval entry points are unsupported")
    return module, targets[0]


def _humaneval_signature(prompt: str, entry_point: Optional[str] = None) -> str:
    """Extract the declared target's interface; preserve simple source headers."""
    _, target = _humaneval_target(prompt, entry_point)
    line = prompt.splitlines()[target.lineno - 1].strip()
    if line.startswith("def ") and line.endswith(":"):
        return line[len("def "):-1]
    signature = f"{target.name}({ast.unparse(target.args)})"
    if target.returns is not None:
        signature += f" -> {ast.unparse(target.returns)}"
    return signature


_HUMANEVAL_SPECIFICATION_EXCLUSIONS = {
    "HumanEval/32": (
        "specification permits multiple valid roots; comparison with one frozen "
        "reference value cannot grade specification-to-output prediction"
    ),
    "HumanEval/50": (
        "insufficient standalone decoding specification; generated inputs can "
        "fall outside the public encoder's lowercase output domain"
    ),
}

# Reviewed public helper, never executed or sent to the model. Parse both the
# actual and reviewed helper on the same interpreter: AST formatting varies
# across supported Python versions. Changes require specification review.
_CYCLIC_PUBLIC_HELPER_SOURCE = '''
def encode_cyclic(s: str):
    """returns encoded string by cycling groups of three characters."""
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]
    return "".join(groups)
'''
_CYCLIC_PUBLIC_HELPER_SPEC = (
    "Public helper specification (from the task prompt):\n"
    "encode_cyclic partitions the input string from left to right into "
    "consecutive groups of three Python characters. In each full "
    "three-character group it moves the first character to the end. "
    "A final group of one or two characters is unchanged. The groups are "
    "joined without separators. Decoding reverses this transformation: "
    "move the last character of each full three-character group to its "
    "front, leaving a final shorter group unchanged."
)


def _humaneval_cyclic_helper_spec(module: ast.Module) -> str:
    helpers = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == "encode_cyclic"]
    if len(helpers) != 1 or isinstance(helpers[0], ast.AsyncFunctionDef):
        raise ValueError("decode_cyclic requires its unambiguous public encoder specification")
    def structure(helper):
        docstring = ast.get_docstring(helper)
        return {
            "name": helper.name,
            "arguments": ast.dump(helper.args, include_attributes=False),
            "docstring": docstring,
            "body": [ast.dump(node, include_attributes=False)
                     for node in (helper.body[1:] if docstring is not None else helper.body)],
            "returns": ast.dump(helper.returns, include_attributes=False) if helper.returns else None,
            "decorators": [ast.dump(node, include_attributes=False) for node in helper.decorator_list],
            "type_parameters": [ast.dump(node, include_attributes=False) for node in getattr(helper, "type_params", [])],
        }

    reviewed = ast.parse(_CYCLIC_PUBLIC_HELPER_SOURCE).body[0]
    if structure(helpers[0]) != structure(reviewed):
        raise ValueError("decode_cyclic public helper changed; its declarative specification requires review")
    return _CYCLIC_PUBLIC_HELPER_SPEC


def _humaneval_docstring(prompt: str, *, keep_doctests: bool,
                        entry_point: Optional[str] = None) -> str:
    """Extract the declared entry point's description, optionally retaining examples.

    Ignore module/helper docstrings. decode_cyclic adds reviewed public-helper
    context; implementation code is neither executed nor included."""
    module, target = _humaneval_target(prompt, entry_point)
    body = ast.get_docstring(target, clean=False)
    if body is None or not body.strip():
        raise ValueError("HumanEval entry point has no nonempty specification docstring")
    lines = []
    for line in body.splitlines():
        if not keep_doctests and line.strip().startswith(">>>"):
            break
        lines.append(line)
    text = "\n".join(lines)
    # Collapse the indentation the prompt carries and tidy blank edges.
    text = "\n".join(l.strip() for l in text.splitlines())
    description = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not description:
        raise ValueError("HumanEval entry point has no standalone description before its examples")
    if target.name == "decode_cyclic":
        description += "\n\n" + _humaneval_cyclic_helper_spec(module)
    return description


def _humaneval_pairs(test_src: str) -> Tuple[List[Any], List[Any]]:
    """Read aligned literal inputs/results lists; reject missing or nonliteral targets."""
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
    """Match base inputs by content; dataset/package ordering may differ.

    Return None when metadata is unavailable so callers record unknown pool labels."""
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
    """Load base/plus invocations and literal targets, retaining flagged ineligible tasks."""
    from datasets import load_dataset

    base_keys = _humaneval_base_keys()
    items: List[SpecItem] = []
    for row in load_dataset(HUMANEVALPLUS_DATASET, split="test"):
        task_id = row["task_id"]
        try:
            signature = _humaneval_signature(row["prompt"], row["entry_point"])
            # Reject unusable signatures before they enter the selection pool.
            parse_signature(signature)
            docstring = _humaneval_docstring(row["prompt"], keep_doctests=keep_doctests,
                                             entry_point=row["entry_point"])
        except (ValueError, SyntaxError) as exc:
            items.append(SpecItem(task_id, "", "", row["entry_point"],
                                  ineligible_reason=str(exc)))
            continue
        if task_id in _HUMANEVAL_SPECIFICATION_EXCLUSIONS:
            items.append(SpecItem(task_id, signature, docstring, row["entry_point"],
                                  ineligible_reason=_HUMANEVAL_SPECIFICATION_EXCLUSIONS[task_id]))
            continue
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


# MBPP+

def _mbpp_signature(code: str, entry_point: Optional[str] = None) -> Tuple[str, str]:
    """Extract MBPP's function name and signature; never expose its implementation."""
    # Dataset regex strings can trigger escape warnings on Python 3.12+.
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


def load_mbppplus() -> List[SpecItem]:
    """Prefer published targets; missing lists may use public base asserts.

    Malformed published lists fail closed. Reference implementations never run.
    """
    from datasets import load_dataset
    from mbpp_published import extract_pairs, NoPublishedTargets, base_input_keys, pool_for

    items: List[SpecItem] = []
    for row in load_dataset(MBPPPLUS_DATASET, split="test"):
        item_id = f"Mbpp/{row['task_id']}"
        try:
            signature, entry_point = _mbpp_signature(row["code"])
        except (ValueError, SyntaxError) as exc:
            items.append(SpecItem(item_id, "", row["prompt"], "",
                                  ineligible_reason=f"unparseable reference: {exc}"))
            continue
        if item_id in _MBPP_SPECIFICATION_EXCLUSIONS:
            items.append(SpecItem(item_id, signature, (row["prompt"] or "").strip(), entry_point,
                                  ineligible_reason=_MBPP_SPECIFICATION_EXCLUSIONS[item_id]))
            continue
        names = _param_names(signature)
        try:
            if not row.get("test"):
                raise NoPublishedTargets("source has no test program")
            published = extract_pairs(row["test"], entry_point, len(names))
        except NoPublishedTargets:
            pass
        except (ValueError, SyntaxError) as exc:
            items.append(SpecItem(item_id, signature, (row["prompt"] or "").strip(), entry_point,
                                  ineligible_reason=f"unusable published ground truth: {exc}"))
            continue
        else:
            base_keys = base_input_keys(row.get("test_list") or [], entry_point)
            invocations = [Invocation(
                invocation_id=f"published:{index}", inputs=_to_kwargs(values, names),
                expected=target, input_line=_render_call(entry_point, values),
                pool=pool_for(values, base_keys),
            ) for index, values, target in published.pairs]
            reason = None if invocations else "no supported published input/output literals"
            items.append(SpecItem(item_id, signature, (row["prompt"] or "").strip(),
                                  entry_point, invocations, ineligible_reason=reason))
            continue
        invocations = _mbpp_invocations(row["test_list"] or [], entry_point, names)
        reason = None if invocations else "no canonical `assert f(...) == literal` test"
        items.append(SpecItem(item_id, signature, (row["prompt"] or "").strip(),
                              entry_point, invocations, ineligible_reason=reason))
    return items


LOADERS = {
    "humaneval_plus": load_humanevalplus,
    "mbpp_plus": load_mbppplus,
}
