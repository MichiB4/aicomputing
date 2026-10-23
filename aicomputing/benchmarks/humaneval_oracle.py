#!/usr/bin/env python3
"""
HumanEval oracle benchmark for AI Computing.

Parses each HumanEval prompt into:
  - signature:  func_name(params) -> type
  - docstring:  the task description (before >>> examples)
  - examples:   list of (inputs_dict, expected_result) from >>> lines

The LLM receives ONLY the signature + docstring + inputs.
It never sees the expected result — pure spec-only prediction.

Usage:
    python benchmarks/humaneval_oracle.py --model qwen3:4b --strategy direct --limit 10
    python benchmarks/humaneval_oracle.py --backend openai --model llama-3.3-70b-versatile \
        --base-url https://api.groq.com/openai/v1 --api-key gsk_xxx --strategy direct

NOTE (revised experimental design)
----------------------------------
This script is no longer the experiment entry point. `--limit N` takes the
first N tasks in problem-file order. Paired comparisons require matching
problem files, task ordering and limits; this entry point records no manifest.

It retains this benchmark's grader (imported by benchmarks/adapters.py) and
an ad-hoc entry point. Historical pilot reproduction requires the matching
code revision and manifest.

For the experiment use:
    python scripts/run_experiment.py --manifest manifests/main-v2.json ...
See aicomputing/README.md for manifest configuration and execution.
"""

from __future__ import annotations

import argparse
import ast
import csv
import doctest
import json
import math
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from bench_args import (  # adjusts sys.path first
    add_common_args, resolve_args, build_runtime, execute_resilient,
    configure_system_certificates,
)

# Configure application HTTPS before importing clients; grader imports stay inert.
if __name__ == "__main__":
    configure_system_certificates()

from aicomputing import VirtualRuntime, ExecuteRequest

# Imported lazily rather than at module scope: a missing optional dependency
# must not kill the process while pytest is merely collecting this file, and
# adapters.py imports this module for results_match() alone. The failure is
# raised when the dataset is actually needed.
def read_problems(*args, **kwargs):
    try:
        from human_eval.data import read_problems as _read
    except ImportError as exc:
        raise ImportError(
            "this benchmark needs human-eval: pip install human-eval"
        ) from exc
    return _read(*args, **kwargs)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it


# ---------------------------------------------------------------------------
# Prompt parser — extracts signature, docstring, examples from HumanEval
# ---------------------------------------------------------------------------

def parse_prompt(prompt: str) -> Tuple[str, str, List[Dict]]:
    """
    Parse a HumanEval prompt into its components.

    Returns:
        signature: "func_name(params) -> type"
        docstring: task description only (no examples)
        examples:  [{"input_line": "func(a, b)", "inputs": {...}, "expected": value}, ...]
    """
    signature = _extract_signature(prompt)
    func_name = signature.split("(")[0]
    param_names = _extract_param_names(signature)

    # Extract full docstring lines
    doc_lines, example_blocks = _split_docstring(prompt)
    docstring = "\n".join(doc_lines).strip()

    # Parse each >>> example
    examples = []
    for call_line, result_line in example_blocks:
        inputs = _parse_call_inputs(call_line, func_name, param_names)
        expected = _parse_expected(result_line)
        examples.append({
            "input_line": call_line,
            "expected_line": result_line,
            "inputs": inputs,
            "expected": expected,
        })

    return signature, docstring, examples


def _extract_signature(prompt: str) -> str:
    for line in prompt.strip().splitlines():
        line = line.strip()
        if line.startswith("def "):
            return line[4:].rstrip(":").strip()
    return "f(x)"


def _extract_param_names(signature: str) -> List[str]:
    """Read positional names without evaluating annotations or defaults."""
    try:
        args = ast.parse(f"def {signature}:\n    pass").body[0].args
    except SyntaxError:
        return []
    return [arg.arg for arg in (*args.posonlyargs, *args.args)]


def _split_docstring(prompt: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Split docstring into description lines and (call, result) pairs.

    Returns:
        doc_lines: lines before the first >>>
        example_blocks: [(call_line, result_line), ...]
    """
    lines = prompt.strip().splitlines()
    in_doc = False
    raw_doc = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if in_doc:
                rest = stripped[3:]
                if rest:
                    raw_doc.append(rest)
                break
            else:
                in_doc = True
                rest = stripped[3:]
                if rest.endswith('"""') or rest.endswith("'''"):
                    raw_doc.append(rest[:-3])
                    break
                if rest:
                    raw_doc.append(rest)
        elif in_doc:
            raw_doc.append(stripped)

    # Preserve complete doctest blocks while keeping only preceding prose.
    parts = doctest.DocTestParser().parse("\n".join(raw_doc))
    doc_lines = parts[0].splitlines()
    example_blocks = [
        (part.source.strip(), part.want.rstrip("\n"))
        for part in parts
        if isinstance(part, doctest.Example)
    ]
    return doc_lines, example_blocks


def _parse_call_inputs(call_line: str, func_name: str, param_names: List[str]) -> Dict[str, Any]:
    """
    Parse literal positional and named arguments, retaining raw fallbacks.
    """
    # Extract the arguments portion: everything inside the outermost parens
    m = re.match(rf"{re.escape(func_name)}\s*\((.*)\)\s*$", call_line, re.DOTALL)
    if not m:
        return {"input": call_line}

    args_str = m.group(1).strip()
    if not args_str:
        return {}

    # Parse literal positional and named arguments.
    keywords = {}
    try:
        # Wrap in a function call to parse: f(arg1, arg2)
        tree = ast.parse(f"f({args_str})", mode="eval")
        call_node = tree.body
        if isinstance(call_node, ast.Call):
            args = [ast.literal_eval(a) for a in call_node.args]
            for keyword in call_node.keywords:
                if keyword.arg is None or keyword.arg in keywords:
                    return {"input": args_str}
                keywords[keyword.arg] = ast.literal_eval(keyword.value)
        else:
            args = [ast.literal_eval(tree.body)]
    except (ValueError, SyntaxError):
        keywords = {}
        # Fallback: try literal_eval on the whole thing
        try:
            val = ast.literal_eval(args_str)
            args = [val] if not isinstance(val, tuple) else list(val)
        except (ValueError, SyntaxError):
            return {"input": args_str}

    inputs = {}
    for i, val in enumerate(args):
        if i < len(param_names):
            inputs[param_names[i]] = val
        else:
            inputs[f"arg{i}"] = val
    if inputs.keys() & keywords.keys():
        return {"input": args_str}
    inputs.update(keywords)
    return inputs


def _parse_expected(result_line: str) -> Any:
    """Parse the expected result line into a Python value."""
    if not result_line:
        return None
    try:
        return ast.literal_eval(result_line)
    except (ValueError, SyntaxError):
        return result_line


def _unordered_results_match(predicted: List[Any], expected: List[Any], tolerance: float) -> bool:
    """Find a one-to-one match without depending on container iteration order."""
    if len(predicted) != len(expected):
        return False
    candidates = [
        [index for index, value in enumerate(expected)
         if results_match(item, value, tolerance)]
        for item in predicted
    ]
    matches: List[Optional[int]] = [None] * len(expected)

    def assign(item_index: int, visited: set) -> bool:
        for index in candidates[item_index]:
            if index in visited:
                continue
            visited.add(index)
            previous = matches[index]
            if previous is None or assign(previous, visited):
                matches[index] = item_index
                return True
        return False

    return all(assign(index, set()) for index in range(len(predicted)))


def results_match(predicted: Any, expected: Any, tolerance: float = 1e-6) -> bool:
    """Compare values recursively, without stringification or boolean coercion.

    Integers and floats may match numerically using the absolute tolerance;
    list/tuple sequences may match across those two container types. Dictionary
    entries and set/frozenset members match one-to-one without regard to order.
    Strings retain case and whitespace. Nonfinite floats never match.
    Other value types require matching types and their normal equality.
    """
    if predicted is None or expected is None:
        return predicted is expected
    if isinstance(predicted, bool) or isinstance(expected, bool):
        return isinstance(predicted, bool) and isinstance(expected, bool) and predicted == expected
    if isinstance(predicted, (int, float)) or isinstance(expected, (int, float)):
        if not isinstance(predicted, (int, float)) or not isinstance(expected, (int, float)):
            return False
        if any(isinstance(value, float) and not math.isfinite(value)
               for value in (predicted, expected)):
            return False
        if predicted == expected:
            return True
        try:
            return abs(predicted - expected) < tolerance
        except OverflowError:
            return False
    if isinstance(predicted, (list, tuple)) or isinstance(expected, (list, tuple)):
        return (
            isinstance(predicted, (list, tuple)) and isinstance(expected, (list, tuple))
            and len(predicted) == len(expected)
            and all(results_match(left, right, tolerance) for left, right in zip(predicted, expected))
        )
    if isinstance(predicted, dict) or isinstance(expected, dict):
        return (
            isinstance(predicted, dict) and isinstance(expected, dict)
            and _unordered_results_match(list(predicted.items()), list(expected.items()), tolerance)
        )
    if isinstance(predicted, (set, frozenset)) or isinstance(expected, (set, frozenset)):
        return (
            isinstance(predicted, (set, frozenset)) and isinstance(expected, (set, frozenset))
            and _unordered_results_match(list(predicted), list(expected), tolerance)
        )
    return type(predicted) is type(expected) and bool(predicted == expected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="HumanEval oracle benchmark")
    add_common_args(ap)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--problem-file", default=None)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()
    if args.limit is not None and args.limit < 0:
        ap.error("--limit must be non-negative")
    cfg = resolve_args(args)

    if args.problem_file:
        problems = read_problems(args.problem_file)
    else:
        from human_eval.data import HUMAN_EVAL   # lazy: see read_problems above
        problems = read_problems(HUMAN_EVAL)
    task_ids = list(problems.keys())
    if args.limit is not None:
        task_ids = task_ids[:args.limit]

    vr = build_runtime(cfg)

    print(f"Paradigm: AI COMPUTING (spec-only prediction)")
    print(f"Tasks:    {len(task_ids)}")
    print()

    all_results = []
    task_pass_count = 0

    def _json(value):
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            return json.dumps(str(value))

    for tid in tqdm(task_ids, desc="Tasks"):
        task = problems[tid]
        signature, docstring, examples = parse_prompt(task["prompt"])

        if not examples:
            all_results.append({
                "task_id": tid, "example_idx": 0,
                "signature": signature, "docstring": docstring,
                "inputs": {}, "input_line": "",
                "raw_text": "", "parsed_result": None,
                "expected": None, "expected_line": "",
                "correct": False, "latency_ms": 0, "tokens_used": 0, "token_retries": 0,
                "benchmark": "humaneval", "type_warnings": "[]",
                "exec_status": "", "pre_revision_correct": "",
                "revised": None, "pre_revision_value": "null",
                "prompts": "[]",
                "reasoning_requests": "[]",
            })
            continue

        task_ok = True
        for i, ex in enumerate(examples):

            r = execute_resilient(vr, ExecuteRequest(
                signature=signature,
                docstring=docstring,
                inputs=ex["inputs"],
                strategy=cfg.strategy,
                strict_types=False,
                max_new_tokens=cfg.max_tokens,
            ), label=f"{tid}#{i}")

            correct = (
                r.exec_status in (None, "ok")
                and r.validation_status != "unparsed"
                and results_match(r.result, ex["expected"])
            )
            pre_rev_correct = (
                "" if r.revised is None
                else results_match(r.pre_revision_value, ex["expected"])
            )
            if not correct:
                task_ok = False

            all_results.append({
                "task_id": tid,
                "example_idx": i,
                "signature": signature,
                "docstring": docstring,
                "inputs": ex["inputs"],
                "input_line": ex["input_line"],
                "raw_text": r.raw_text,
                "parsed_result": r.result,
                "expected": ex["expected"],
                "expected_line": ex["expected_line"],
                "correct": correct,
                "latency_ms": r.latency_ms,
                "token_retries": r.token_retries,
                "benchmark": "humaneval",
                "type_warnings": json.dumps(r.type_warnings, default=str),
                "exec_status": r.exec_status or "",
                "pre_revision_correct": pre_rev_correct,
                "revised": r.revised,
                "pre_revision_value": _json(r.pre_revision_value),
                "tokens_used": r.tokens_used,
                "prompts": json.dumps(r.prompts, default=str),
                "reasoning_requests": json.dumps(r.reasoning_requests, default=str),
            })

        if task_ok:
            task_pass_count += 1

    # Summary
    total_tasks = len(task_ids)
    total_examples = len(all_results)
    correct_examples = sum(1 for r in all_results if r["correct"])
    task_accuracy = task_pass_count / total_tasks if total_tasks else 0.0
    example_accuracy = correct_examples / total_examples if total_examples else 0.0

    print(f"\n{'='*50}")
    print(f"Tasks passed:    {task_pass_count}/{total_tasks} ({task_accuracy:.4f})")
    print(f"Examples correct: {correct_examples}/{total_examples} ({example_accuracy:.4f})")
    print()

    # Print details
    current_task = None
    for r in all_results[:30]:
        if r["task_id"] != current_task:
            current_task = r["task_id"]
            # Check if all examples for this task are correct
            task_correct = all(x["correct"] for x in all_results if x["task_id"] == current_task)
            tag = "PASS" if task_correct else "FAIL"
            print(f"\n  {current_task}: {tag}")
        mark = "✓" if r["correct"] else "✗"
        print(f"    {mark} {r['input_line'][:60]}")
        print(f"      expected: {r['expected_line']}")
        print(f"      got:      {r['raw_text'][:80]}")

    # Write CSV
    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "task_id", "example_idx", "correct",
                "signature", "docstring",
                "input_line", "inputs",
                "expected_line", "expected",
                "raw_text", "parsed_result",
                "prompts",
                "model", "strategy",
                "latency_ms", "tokens_used", "token_retries",
                "benchmark", "type_warnings", "exec_status",
                "pre_revision_correct", "revised", "pre_revision_value",
                "reasoning_requests",
            ])
            for r in all_results:
                w.writerow([
                    r["task_id"], r["example_idx"], r["correct"],
                    r["signature"], r["docstring"],
                    r["input_line"], _json(r["inputs"]),
                    r["expected_line"], _json(r["expected"]),
                    r["raw_text"], _json(r["parsed_result"]),
                    r["prompts"],
                    cfg.model, cfg.strategy,
                    f"{r['latency_ms']:.0f}", r["tokens_used"], r["token_retries"],
                    r["benchmark"], r["type_warnings"], r["exec_status"],
                    r["pre_revision_correct"], r["revised"], r["pre_revision_value"],
                    r["reasoning_requests"],
                ])
        print(f"\nResults written to {args.out_csv}")


if __name__ == "__main__":
    main()
