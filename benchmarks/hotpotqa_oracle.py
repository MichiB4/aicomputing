#!/usr/bin/env python3
"""
HotpotQA oracle benchmark for AI Computing.

Tests compositional multi-step reasoning — the model must answer
questions that require chaining information from multiple sources.
We provide the supporting context so the test is about reasoning,
not retrieval.

Dataset: hotpotqa/hotpot_qa (distractor setting, validation split)
Each problem has a question, supporting context paragraphs, and answer.

Usage:
    python benchmarks/hotpotqa_oracle.py --model qwen3:4b --strategy direct --limit 50
    python benchmarks/hotpotqa_oracle.py --backend openai --model llama-3.3-70b-versatile \
        --base-url https://api.groq.com/openai/v1 --api-key gsk_xxx --limit 100
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
import sys
from typing import Any, Dict, List, Optional

from aicomputing import VirtualRuntime, ExecuteRequest
from bench_args import add_common_args, resolve_args
from aicomputing.prompting import build_direct_prompt, build_cot_prompt

try:
    from datasets import load_dataset
except ImportError:
    print("Install datasets: pip install datasets", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it


def build_context(row: Dict) -> str:
    """Build a readable context string from HotpotQA context field."""
    titles = row["context"]["title"]
    sentences_list = row["context"]["sentences"]
    parts = []
    for title, sentences in zip(titles, sentences_list):
        text = " ".join(sentences)
        parts.append(f"[{title}] {text}")
    # Limit to ~2000 chars to fit in prompt
    context = "\n".join(parts)
    return context


def normalize_answer(s: str) -> str:
    """Normalize answer for F1/EM comparison (from SQuAD eval script)."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(normalize_answer(prediction) == normalize_answer(ground_truth))
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Exact match after normalization."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def main():
    ap = argparse.ArgumentParser(description="HotpotQA oracle benchmark")
    add_common_args(ap)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()
    cfg = resolve_args(args)

    think_override = cfg.think_override

    print("Loading HotpotQA dataset...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    vr = VirtualRuntime(
        backend=cfg.backend, model=cfg.model,
        strategy=cfg.strategy, trace_file=cfg.trace,
        **cfg.backend_kwargs,
    )

    print(f"Benchmark: HotpotQA ({len(ds)} problems)")
    print()

    _SIGNATURE = "answer_question(context: str, question: str) -> str"
    _DOCSTRING = (
        "Given the context paragraphs, answer the question. "
        "Return ONLY the answer — a short phrase, name, date, or yes/no. "
        "Do not explain."
    )

    results = []
    n_em = 0
    f1_sum = 0.0

    for i, row in enumerate(tqdm(ds, desc="HotpotQA")):
        question = row["question"]
        expected = row["answer"]
        context = build_context(row)
        q_type = row.get("type", "unknown")

        r = vr.execute(ExecuteRequest(
            signature=_SIGNATURE,
            docstring=_DOCSTRING,
            inputs={"context": context, "question": question},
            return_type=str,
            strategy=cfg.strategy,
            think=think_override,
            strict_types=False,
        ))

        predicted = str(r.result) if r.result is not None else ""
        em = exact_match(predicted, expected)
        f1 = compute_f1(predicted, expected)
        if em: n_em += 1
        f1_sum += f1

        prompt_sent = build_cot_prompt(_SIGNATURE, _DOCSTRING, {"context": context, "question": question}) \
            if cfg.strategy == "cot" else \
            build_direct_prompt(_SIGNATURE, _DOCSTRING, {"context": context, "question": question})

        results.append({
            "task_id": f"hotpotqa/{i}",
            "example_idx": 0,
            "correct": em,
            "signature": _SIGNATURE,
            "docstring": _DOCSTRING,
            "input_line": question[:150],
            "inputs": json.dumps({"context": context, "question": question}, ensure_ascii=False),
            "expected_line": expected,
            "expected": json.dumps(expected),
            "raw_text": r.raw_text,
            "parsed_result": json.dumps(predicted),
            "prompt": prompt_sent,
            "model": cfg.model,
            "strategy": r.strategy,
            "latency_ms": r.latency_ms,
            "token_retries": r.token_retries,
            "tokens_used": r.tokens_used,
            "f1": f1,
            "type": q_type,
        })

    total = len(results)
    print(f"\n{'='*50}")
    print(f"HotpotQA EM:  {n_em}/{total} ({n_em/total:.4f})")
    print(f"HotpotQA F1:  {f1_sum/total:.4f}")
    print()
    for r in results[:10]:
        mark = "✓" if r["correct"] else "✗"
        print(f"  {mark} [{r['type']}] expected={r['expected_line']!r}, got={r['parsed_result']!r} (F1={r['f1']:.2f})")

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
                "f1", "type",
            ])
            for r in results:
                w.writerow([
                    r["task_id"], r["example_idx"], r["correct"],
                    r["signature"], r["docstring"],
                    r["input_line"], r["inputs"],
                    r["expected_line"], r["expected"],
                    r["raw_text"], r["parsed_result"],
                    r["prompt"],
                    r["model"], r["strategy"],
                    f"{r['latency_ms']:.0f}", r["tokens_used"], r["token_retries"],
                    f"{r['f1']:.4f}", r["type"],
                ])
        print(f"Results written to {args.out_csv}")

if __name__ == "__main__":
    main()
