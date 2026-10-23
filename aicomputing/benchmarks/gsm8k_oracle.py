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
This script is no longer the experiment entry point. `--limit N` takes the
first N problems in dataset order. Paired comparisons require matching data,
ordering and limits; this entry point records no manifest.

It retains this benchmark's grader (imported by benchmarks/adapters.py) and
an ad-hoc entry point. Historical pilot reproduction requires the matching
code revision and manifest.

For the experiment use:
    python scripts/run_experiment.py --manifest manifests/main-v2.json ...
See aicomputing/README.md for manifest configuration and execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from typing import Any, Dict, List, Optional

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


_NUMBER_PATTERN = re.compile(
    r"[+-]?(?:(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]*)?|\.[0-9]+)"
    r"(?:[eE][+-]?[0-9]+)?"
)


def _finite_number(value: Any) -> Optional[float]:
    """Read a finite number or wholly numeric text, never a boolean or prose."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        if _NUMBER_PATTERN.fullmatch(text) is None:
            return None
        value = text.replace(",", "")
    elif not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def extract_gsm8k_answer(answer_str: str) -> float:
    """Read the numeric target following one explicit #### final-answer marker."""
    if not isinstance(answer_str, str) or answer_str.count("####") != 1:
        raise ValueError("GSM8K ground truth requires one #### final-answer marker")
    number = _finite_number(answer_str.split("####", 1)[1])
    if number is None:
        raise ValueError("GSM8K #### final answer must be wholly numeric and finite")
    return number


def answers_match(predicted: Any, expected: float, tolerance: float = 1e-3) -> bool:
    """Compare finite numeric answers using the existing absolute tolerance."""
    predicted_number = _finite_number(predicted)
    expected_number = _finite_number(expected)
    if predicted_number is None or expected_number is None:
        return False
    return abs(predicted_number - expected_number) < tolerance


def main():
    ap = argparse.ArgumentParser(description="GSM8K oracle benchmark")
    add_common_args(ap)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()
    if args.limit is not None and args.limit < 0:
        ap.error("--limit must be non-negative")
    cfg = resolve_args(args)

    print("Loading GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.limit is not None:
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
            strict_types=False,
            max_new_tokens=cfg.max_tokens,
        ), label=f"gsm8k#{i}")

        correct = r.exec_status in (None, "ok") and answers_match(r.result, expected)
        pre_rev_correct = (
            "" if r.revised is None
            else answers_match(r.pre_revision_value, expected)
        )
        if correct: n_correct += 1

        serialized = {}
        for field, value in (("parsed_result", r.result),
                             ("pre_revision_value", r.pre_revision_value)):
            try:
                serialized[field] = json.dumps(value, default=str)
            except (TypeError, ValueError):
                serialized[field] = json.dumps(str(value))

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
            "parsed_result": serialized["parsed_result"],
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
            "pre_revision_value": serialized["pre_revision_value"],
            "tokens_used": r.tokens_used,
            "reasoning_requests": json.dumps(r.reasoning_requests, default=str),
        })

    total = len(results)
    accuracy = n_correct / total if total else 0.0
    print(f"\n{'='*50}")
    print(f"GSM8K Accuracy: {n_correct}/{total} ({accuracy:.4f})")
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
                "reasoning_requests",
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
                    r["reasoning_requests"],
                ])
        print(f"Results written to {args.out_csv}")

if __name__ == "__main__":
    main()
