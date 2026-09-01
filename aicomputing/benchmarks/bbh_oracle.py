#!/usr/bin/env python3
"""
BIG-Bench Hard oracle benchmark for AI Computing.

Tests deterministic logic and pattern-following — the "easiest" case
for oracle prediction since the functions are well-defined and simple.

Dataset: lukaemon/bbh (23 tasks, 250 examples each)
Tasks used:
  - date_understanding: date reasoning and calculation
  - boolean_expressions: evaluate True/False expressions
  - tracking_shuffled_objects_three_objects: follow object swaps
  - logical_deduction_five_objects: deduce order from clues
  - web_of_lies: truth/lie chain reasoning

Usage:
    python benchmarks/bbh_oracle.py --model qwen3:4b --strategy direct --limit 50
    python benchmarks/bbh_oracle.py --backend openai --model llama-3.3-70b-versatile \
        --base-url https://api.groq.com/openai/v1 --api-key gsk_xxx

NOTE (revised experimental design)
----------------------------------
This script is no longer the experiment entry point. It samples with
`--limit N`, unseeded and without a manifest, so different runs and different
models can see different items -- which breaks the paired analysis.

It is retained because it holds this benchmark's grader (imported by
benchmarks/adapters.py) and because it reproduces the historical pilot results.

For the experiment use:
    python scripts/run_experiment.py --manifest manifests/main-v1.json ...
See CHANGES_EXPERIMENT.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from typing import Any, Dict, List, Optional

from bench_args import (  # adjusts sys.path first
    add_common_args, resolve_args, build_runtime, execute_resilient,
)
from aicomputing import VirtualRuntime, ExecuteRequest

# Imported lazily rather than at module scope: a missing optional dependency
# must not kill the process while pytest is merely collecting this file, and
# adapters.py imports these modules for their graders alone. The failure is
# raised when the dataset is actually needed.
def load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset as _load
    except ImportError as exc:
        raise ImportError(
            "this benchmark needs the `datasets` package: pip install datasets"
        ) from exc
    return _load(*args, **kwargs)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it


# Tasks to evaluate — these cover logic, dates, and reasoning
DEFAULT_TASKS = [
    "date_understanding",
    "boolean_expressions",
    "tracking_shuffled_objects_three_objects",
    "logical_deduction_five_objects",
    "web_of_lies",
]

# Map task names to AI function signatures and docstrings
TASK_SPECS = {
    "date_understanding": {
        "signature": "answer_date_question(question: str) -> str",
        "docstring": "Answer the date reasoning question. Return ONLY the answer option label, including its parentheses, e.g. '(A)'.",
    },
    "boolean_expressions": {
        "signature": "evaluate_boolean(expression: str) -> str",
        "docstring": "Evaluate the boolean expression. Return ONLY 'True' or 'False'.",
    },
    "tracking_shuffled_objects_three_objects": {
        "signature": "track_objects(question: str) -> str",
        "docstring": "Track the objects through swaps and answer the question. Return ONLY the answer option label, including its parentheses, e.g. '(A)'.",
    },
    "logical_deduction_five_objects": {
        "signature": "deduce_order(question: str) -> str",
        "docstring": "Deduce the order from the clues and answer. Return ONLY the answer option label, including its parentheses, e.g. '(A)'.",
    },
    "web_of_lies": {
        "signature": "evaluate_truth(question: str) -> str",
        "docstring": "Follow the chain of truth-tellers and liars. Return ONLY 'Yes' or 'No'.",
    },
}


def normalize_answer(text: str) -> str:
    """
    Normalize an answer for comparison.

    Multiple-choice labels are folded to the parenthesised form "(a)", which
    is how BBH writes its gold answers. Accepting the bare form matters: the
    gold label is "(B)" but a model asked for "the answer option" will often
    reply "B", and scoring that wrong measures formatting compliance rather
    than reasoning. It cost two whole BBH tasks (0/5 each, both arms) on the
    first Groq pilot before the spec was clarified to name the format.

    Folding is one-directional and only fires on a lone letter or a letter
    with trailing punctuation, so it cannot collide with the free-text tasks
    here — those answer with dates, "True"/"False", or "Yes"/"No", none of
    which is a single character.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    # Strip a leading "answer:" / "the answer is" preamble before matching.
    text = re.sub(r"^(?:the\s+)?answer(?:\s+is)?\s*:?\s*", "", text)
    text = text.strip().strip("\"'")
    # Extract option label if present: "(a)", "(b)", etc.
    m = re.search(r"\(([a-z])\)", text)
    if m:
        return f"({m.group(1)})"
    # Bare label: "b", "b)", "b." — but never a boolean word, handled below.
    m = re.fullmatch(r"([a-z])[).:]?", text)
    if m:
        return f"({m.group(1)})"
    # Normalize boolean
    if text in ("true", "yes"):
        return "true"
    if text in ("false", "no"):
        return "false"
    return text


def main():
    ap = argparse.ArgumentParser(description="BIG-Bench Hard oracle benchmark")
    add_common_args(ap)
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                    help="BBH tasks to evaluate")
    ap.add_argument("--limit", type=int, default=None, help="Limit per task")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()
    cfg = resolve_args(args)

    think_override = cfg.think_override

    vr = build_runtime(cfg)

    print(f"Benchmark: BIG-Bench Hard")
    print(f"Tasks:     {args.tasks}")
    print()

    all_results = []

    for task_name in args.tasks:
        print(f"--- {task_name} ---")
        try:
            ds = load_dataset("lukaemon/bbh", task_name, split="test")
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))

        spec = TASK_SPECS.get(task_name, {
            "signature": f"answer(question: str) -> str",
            "docstring": "Answer the question. Return ONLY the answer.",
        })

        n_correct = 0
        task_results = []

        for i, row in enumerate(tqdm(ds, desc=task_name)):
            question = row["input"]
            expected = row["target"]

            r = execute_resilient(vr, ExecuteRequest(
                signature=spec["signature"],
                docstring=spec["docstring"],
                inputs={"question": question},
                return_type=str,
                strategy=cfg.strategy,
                think=think_override,
                strict_types=False,
                max_new_tokens=cfg.max_tokens,
            ), label=f"{task_name}#{i}")

            predicted = str(r.result) if r.result is not None else ""
            correct = normalize_answer(predicted) == normalize_answer(expected)
            pre_rev_correct = (
                "" if r.revised is None
                else normalize_answer(str(r.pre_revision_value or "")) == normalize_answer(expected)
            )
            if correct: n_correct += 1


            task_results.append({
                "task_id": f"{task_name}/{i}",
                "example_idx": 0,
                "correct": correct,
                "signature": spec["signature"],
                "docstring": spec["docstring"],
                "input_line": question[:150],
                "inputs": json.dumps({"question": question}, ensure_ascii=False),
                "expected_line": expected,
                "expected": json.dumps(expected),
                "raw_text": r.raw_text,
                "parsed_result": json.dumps(predicted),
                "prompts": json.dumps(r.prompts, default=str),
                "model": cfg.model,
                "strategy": r.strategy,
                "latency_ms": r.latency_ms,
                "token_retries": r.token_retries,
                "benchmark": "bbh",
                "type_warnings": json.dumps(r.type_warnings, default=str),
                "exec_status": r.exec_status or "",
                "pre_revision_correct": pre_rev_correct,
                "revised": r.revised,
                "pre_revision_value": json.dumps(r.pre_revision_value, default=str),
                "tokens_used": r.tokens_used,
            })

        total = len(task_results)
        acc = n_correct / total if total > 0 else 0
        print(f"  Accuracy: {n_correct}/{total} ({acc:.4f})")
        all_results.extend(task_results)

    # Overall
    total_correct = sum(1 for r in all_results if r["correct"])
    total = len(all_results)
    print(f"\n{'='*50}")
    print(f"BBH Overall: {total_correct}/{total} ({total_correct/total:.4f})")
    print()
    for r in all_results[:10]:
        mark = "✓" if r["correct"] else "✗"
        print(f"  {mark} [{r['task_id'].split('/')[0]}] expected={r['expected_line']!r}, got={r['parsed_result']!r}")

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
            ])
            for r in all_results:
                w.writerow([
                    r["task_id"], r["example_idx"], r["correct"],
                    r["signature"], r["docstring"],
                    r["input_line"], r["inputs"],
                    r["expected_line"], r["expected"],
                    r["raw_text"], r["parsed_result"],
                    r["prompts"],
                    r["model"], r["strategy"],
                    f"{r['latency_ms']:.0f}", r["tokens_used"], r["token_retries"],
                    r["benchmark"], r["type_warnings"], r["exec_status"],
                    r["pre_revision_correct"], r["revised"], r["pre_revision_value"],
                ])
        print(f"Results written to {args.out_csv}")

if __name__ == "__main__":
    main()
