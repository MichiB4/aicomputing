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
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from aicomputing import VirtualRuntime, ExecuteRequest
from aicomputing.prompting import build_direct_prompt, build_cot_prompt
from bench_args import add_common_args, resolve_args

# Try importing human_eval for dataset loading
try:
    from human_eval.data import read_problems, HUMAN_EVAL
except ImportError:
    print("human-eval not installed. Run: pip install human-eval", file=sys.stderr)
    sys.exit(1)

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
    m = re.match(r"\w+\(([^)]*)\)", signature)
    if not m:
        return []
    params = []
    for part in m.group(1).split(","):
        part = part.strip()
        if part:
            name = part.split(":")[0].split("=")[0].strip()
            params.append(name)
    return params


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

    # Split into description (before >>>) and examples
    doc_lines = []
    example_blocks = []
    i = 0
    while i < len(raw_doc):
        line = raw_doc[i]
        if line.startswith(">>>"):
            # This is a call line; next line is the expected result
            call_line = line[3:].strip()  # strip ">>> "
            result_line = ""
            if i + 1 < len(raw_doc) and not raw_doc[i + 1].startswith(">>>"):
                result_line = raw_doc[i + 1].strip()
                i += 2
            else:
                i += 1
            example_blocks.append((call_line, result_line))
        else:
            doc_lines.append(line)
            i += 1

    return doc_lines, example_blocks


def _parse_call_inputs(call_line: str, func_name: str, param_names: List[str]) -> Dict[str, Any]:
    """
    Parse "func_name(arg1, arg2)" into {"param1": arg1, "param2": arg2}.
    """
    # Extract the arguments portion: everything inside the outermost parens
    m = re.match(rf"{re.escape(func_name)}\s*\((.*)\)\s*$", call_line, re.DOTALL)
    if not m:
        return {"input": call_line}

    args_str = m.group(1).strip()
    if not args_str:
        return {}

    # Parse arguments using ast
    try:
        # Wrap in a function call to parse: f(arg1, arg2)
        tree = ast.parse(f"f({args_str})", mode="eval")
        call_node = tree.body
        if isinstance(call_node, ast.Call):
            args = [ast.literal_eval(a) for a in call_node.args]
        else:
            args = [ast.literal_eval(tree.body)]
    except (ValueError, SyntaxError):
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
    return inputs


def _parse_expected(result_line: str) -> Any:
    """Parse the expected result line into a Python value."""
    if not result_line:
        return None
    try:
        return ast.literal_eval(result_line)
    except (ValueError, SyntaxError):
        return result_line


def results_match(predicted: Any, expected: Any, tolerance: float = 1e-6) -> bool:
    """Compare predicted vs expected with type-aware matching."""
    if predicted is None and expected is not None:
        return False

    # Exact match
    if predicted == expected:
        return True

    # Numeric tolerance
    try:
        if isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
            return abs(float(predicted) - float(expected)) < tolerance
    except (ValueError, TypeError):
        pass

    # List/tuple: compare element-wise
    if isinstance(expected, (list, tuple)) and isinstance(predicted, (list, tuple)):
        if len(expected) != len(predicted):
            return False
        return all(results_match(p, e, tolerance) for p, e in zip(predicted, expected))

    # String comparison (case-insensitive for booleans)
    pred_str = str(predicted).strip().lower()
    exp_str = str(expected).strip().lower()
    return pred_str == exp_str


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
    cfg = resolve_args(args)

    think_override = cfg.think_override

    problems = read_problems(args.problem_file or HUMAN_EVAL)
    task_ids = list(problems.keys())
    if args.limit:
        task_ids = task_ids[:args.limit]

    vr = VirtualRuntime(
        backend=cfg.backend, model=cfg.model,
        strategy=cfg.strategy, trace_file=cfg.trace,
        **cfg.backend_kwargs,
    )

    print(f"Paradigm: AI COMPUTING (spec-only prediction)")
    print(f"Tasks:    {len(task_ids)}")
    print()

    all_results = []
    task_pass_count = 0

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
                "prompt": "",
            })
            continue

        task_ok = True
        for i, ex in enumerate(examples):
            # Build the prompt that the LLM actually sees
            if cfg.strategy == "cot":
                prompt_sent = build_cot_prompt(
                    signature=signature, docstring=docstring,
                    inputs=ex["inputs"],
                )
            else:
                prompt_sent = build_direct_prompt(
                    signature=signature, docstring=docstring,
                    inputs=ex["inputs"],
                )

            r = vr.execute(ExecuteRequest(
                signature=signature,
                docstring=docstring,
                inputs=ex["inputs"],
                strategy=cfg.strategy,
                think=think_override,
                strict_types=False,
            ))

            correct = results_match(r.result, ex["expected"])
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
                "tokens_used": r.tokens_used,
                "prompt": prompt_sent,
            })

        if task_ok:
            task_pass_count += 1

    # Summary
    total_tasks = len(task_ids)
    total_examples = len(all_results)
    correct_examples = sum(1 for r in all_results if r["correct"])

    print(f"\n{'='*50}")
    print(f"Tasks passed:    {task_pass_count}/{total_tasks} ({task_pass_count/total_tasks:.4f})")
    print(f"Examples correct: {correct_examples}/{total_examples} ({correct_examples/total_examples:.4f})")
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
                "prompt",
                "model", "strategy",
                "latency_ms", "tokens_used", "token_retries",
            ])
            for r in all_results:
                w.writerow([
                    r["task_id"], r["example_idx"], r["correct"],
                    r["signature"], r["docstring"],
                    r["input_line"], json.dumps(r["inputs"], default=str),
                    r["expected_line"], json.dumps(r["expected"], default=str),
                    r["raw_text"], json.dumps(r["parsed_result"], default=str),
                    r["prompt"],
                    cfg.model, cfg.strategy,
                    f"{r['latency_ms']:.0f}", r["tokens_used"], r["token_retries"],
                ])
        print(f"\nResults written to {args.out_csv}")


if __name__ == "__main__":
    main()
