#!/usr/bin/env python3
"""
GSM8K oracle benchmark for AI Computing.

Tests numerical precision and multi-step arithmetic — the model must
predict the final numeric answer to grade school math word problems
from the problem description alone (no code execution).

Dataset: openai/gsm8k (1319 test problems)

Usage:
    python benchmarks/gsm8k_oracle.py --model qwen3:4b --strategy direct --limit 50
    python benchmarks/gsm8k_oracle.py --backend openai --model llama-3.3-70b-versatile \
        --base-url https://api.groq.com/openai/v1 --api-key gsk_xxx --limit 100

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
import time
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


def extract_gsm8k_answer(answer_str: str) -> float:
    """Extract the numeric answer after #### from GSM8K answer string."""
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_str)
    if m:
        return float(m.group(1).replace(",", ""))
    nums = re.findall(r"-?[\d,]+\.?\d*", answer_str)
    if nums:
        return float(nums[-1].replace(",", ""))
    return float("nan")


def answers_match(predicted: Any, expected: float, tolerance: float = 1e-3) -> bool:
    """Check if predicted answer matches expected, with numeric tolerance."""
    try:
        pred_num = float(predicted)
    except (ValueError, TypeError):
        if isinstance(predicted, str):
            nums = re.findall(r"-?[\d,]+\.?\d*", predicted)
            if nums:
                pred_num = float(nums[-1].replace(",", ""))
            else:
                return False
        else:
            return False
    return abs(pred_num - expected) < tolerance


def main():
    ap = argparse.ArgumentParser(description="GSM8K oracle benchmark")
    add_common_args(ap)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()
    cfg = resolve_args(args)

    think_override = cfg.think_override

    print("Loading GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    vr = build_runtime(cfg)

    print(f"Benchmark: GSM8K ({len(ds)} problems)")
    print()

    _SIGNATURE = "solve_math(question: str) -> float"
    _DOCSTRING = "Solve the given grade school math word problem. Return ONLY the final numeric answer."

    results = []
    n_correct = 0

    for i, row in enumerate(tqdm(ds, desc="GSM8K")):
        question = row["question"]
        expected = extract_gsm8k_answer(row["answer"])

        r = execute_resilient(vr, ExecuteRequest(
            signature=_SIGNATURE,
            docstring=_DOCSTRING,
            inputs={"question": question},
            return_type=float,
            strategy=cfg.strategy,
            think=think_override,
            strict_types=False,
            max_new_tokens=cfg.max_tokens,
        ), label=f"gsm8k#{i}")

        correct = answers_match(r.result, expected)
        pre_rev_correct = (
            "" if r.revised is None
            else answers_match(r.pre_revision_value, expected)
        )
        if correct: n_correct += 1


        results.append({
            "task_id": f"gsm8k/{i}",
            "example_idx": 0,
            "correct": correct,
            "signature": _SIGNATURE,
            "docstring": _DOCSTRING,
            "input_line": question[:150],
            "inputs": json.dumps({"question": question}, ensure_ascii=False),
            "expected_line": str(expected),
            "expected": json.dumps(expected),
            "raw_text": r.raw_text,
            "parsed_result": json.dumps(r.result),
            "prompts": json.dumps(r.prompts, default=str),
            "model": cfg.model,
            "strategy": r.strategy,
            "latency_ms": r.latency_ms,
            "token_retries": r.token_retries,
            "benchmark": "gsm8k",
            "type_warnings": json.dumps(r.type_warnings, default=str),
            "exec_status": r.exec_status or "",
            "pre_revision_correct": pre_rev_correct,
            "revised": r.revised,
            "pre_revision_value": json.dumps(r.pre_revision_value, default=str),
            "tokens_used": r.tokens_used,
        })

    total = len(results)
    print(f"\n{'='*50}")
    print(f"GSM8K Accuracy: {n_correct}/{total} ({n_correct/total:.4f})")
    print()
    for r in results[:10]:
        mark = "✓" if r["correct"] else "✗"
        print(f"  {mark} expected={r['expected_line']}, got={r['parsed_result']!r}")

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
            for r in results:
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
