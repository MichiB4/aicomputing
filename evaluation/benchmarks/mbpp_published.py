"""Read published MBPP+ data literals without executing its test programs."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple
import warnings

from .values import canonical_json as value_key
from .values import encode_value


class NoPublishedTargets(ValueError):
    """The source requires reference execution instead of listing outputs."""


@dataclass
class PublishedPairs:
    pairs: list[tuple[int, list, Any]] = field(default_factory=list)
    excluded: dict[int, str] = field(default_factory=dict)


def mbpp_signature(code: str) -> Tuple[str, str]:
    """Extract the last function's name and signature; never expose its implementation."""
    # Dataset regex strings can trigger escape warnings on Python 3.12+.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(code)
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not defs:
        raise ValueError("reference solution defines no function")
    fn = defs[-1]
    args = ", ".join(a.arg for a in (*fn.args.posonlyargs, *fn.args.args))
    return f"{fn.name}({args})", fn.name


def param_names(signature: str) -> List[str]:
    """Extract parameter names with AST parsing so generic-annotation commas are safe."""
    args = ast.parse(f"def {signature.rstrip(':')}: pass").body[0].args
    names = [a.arg for a in (*args.posonlyargs, *args.args)]
    if args.vararg:
        names.append(args.vararg.arg)
    return names + [a.arg for a in args.kwonlyargs]


def to_kwargs(values: Sequence[Any], names: Sequence[str]) -> Dict[str, Any]:
    """Zip positional test inputs onto parameter names."""
    if len(values) > len(names):
        raise ValueError("more positional inputs than parameters")
    return dict(zip(names, values))


def extract_pairs(source: str, entry_point: str, arity: int) -> PublishedPairs:
    """Keep aligned, finite, typed literals; record unsupported pairs by index.

    Only the known module-level data layout and direct scoring loop are accepted.
    Imports, function bodies and calls are never evaluated.
    """
    tree = ast.parse(source)
    assignments = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in ("inputs", "results") and name not in assignments:
                assignments[name] = node.value
                continue
        if node is tree.body[-1] and isinstance(node, ast.For):
            continue
        raise ValueError("unreviewed published data layout or reassignment")
    if "results" not in assignments:
        raise NoPublishedTargets("test script has no precomputed results list")
    if set(assignments) != {"inputs", "results"} or any(
            not isinstance(value, ast.List) for value in assignments.values()):
        raise ValueError("published inputs/results must be literal lists")
    inputs, results = assignments["inputs"].elts, assignments["results"].elts
    if not inputs or len(inputs) != len(results):
        raise ValueError("published inputs/results are empty or differ in length")
    loop = tree.body[-1]
    template = ast.parse(
        "for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
        f"    assertion({entry_point}(*inp), exp, 0)\n").body[0]
    # The published numeric tolerance belongs to the scorer, not the data.
    if isinstance(loop, ast.For) and len(loop.body) == 1:
        call = getattr(loop.body[0], "value", None)
        if isinstance(call, ast.Call) and len(call.args) == 3:
            tolerance = call.args[-1]
            if isinstance(tolerance, ast.Constant) and type(tolerance.value) in (int, float):
                template.body[0].value.args[-1] = tolerance
    if ast.dump(loop) != ast.dump(template):
        raise ValueError("unreviewed published input/output scoring loop")
    extracted = PublishedPairs()
    for index, (arguments, expected) in enumerate(zip(inputs, results)):
        try:
            values, target = ast.literal_eval(arguments), ast.literal_eval(expected)
            if type(values) not in (list, tuple) or len(values) != arity:
                raise ValueError("input arity differs from the function signature")
            encode_value(values)
            encode_value(target)
        except (ValueError, TypeError, SyntaxError) as exc:
            # No eval, inf conversion, output coercion, or index renumbering.
            extracted.excluded[index] = str(exc)
            continue
        extracted.pairs.append((index, list(values), target))
    return extracted


def base_input_keys(test_list, entry_point):
    """Match public base calls by typed argument content, including wrapped calls.

    Public assertions are not a complete base catalog across source revisions.
    """
    keys = set()
    for source in test_list:
        try:
            tree = ast.parse(source.strip())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and getattr(node.func, "id", None) == entry_point]
            if len(calls) != 1 or calls[0].keywords:
                return None
            keys.add(value_key([ast.literal_eval(arg) for arg in calls[0].args]))
        except (SyntaxError, ValueError, TypeError):
            return None
    return keys or None


def pool_for(values, base_keys):
    if base_keys is None:
        return "unknown"
    return "base" if value_key(values) in base_keys else "unknown"


_FLOAT_HELPER = '''
def is_floats(x) -> bool:
    if isinstance(x, float): return True
    if isinstance(x, (list, tuple)): return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray): return x.dtype == np.float64 or x.dtype == np.float32
    return False
'''
_ASSERTION = '''
def assertion(out, exp, atol):
    if atol == 0 and is_floats(exp): atol = 1e-6
    if out != exp and atol != 0:
        assert np.allclose(out, exp, rtol=1e-7, atol=atol)
    else:
        assert out == exp, f"out: {out}, exp: {exp}"
'''


def scoring_profile(source):
    """Recognize the two reviewed scoring forms; never execute them."""
    tree = ast.parse(source)
    helpers = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    def same(left, right):
        return left is not None and ast.dump(left) == ast.dump(right)

    if not same(helpers.get("is_floats"), ast.parse(_FLOAT_HELPER).body[0]):
        raise ValueError("unreviewed published float helper")
    ordinary = ast.parse(_ASSERTION).body[0]
    unordered = ast.parse(_ASSERTION.replace(
        "    if out != exp", "    out = set(out)\n    exp = set(exp)\n    if out != exp")).body[0]
    is_unordered = same(helpers.get("assertion"), unordered)
    if not is_unordered and not same(helpers.get("assertion"), ordinary):
        raise ValueError("unreviewed published scoring function")
    loop = tree.body[-1]
    if not isinstance(loop, ast.For) or len(loop.body) != 1:
        raise ValueError("unreviewed published scoring loop")
    statement = loop.body[0]
    call = statement.value if isinstance(statement, ast.Expr) else None
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
            or call.func.id != "assertion" or len(call.args) != 3):
        raise ValueError("unreviewed published scoring loop")
    tolerance = call.args[-1]
    if not isinstance(tolerance, ast.Constant) or type(tolerance.value) not in (int, float):
        raise ValueError("unreviewed published tolerance")
    if tolerance.value not in (0, 0.0001):
        raise ValueError("unreviewed published tolerance")
    if is_unordered:
        if tolerance.value != 0:
            raise ValueError("unreviewed published set tolerance")
        return "unordered_set_abs_1e-6"
    return "mbpp_float_atol_1e-4_v1" if tolerance.value else "mbpp_exact_or_float_v1"
