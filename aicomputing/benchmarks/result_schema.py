"""
The single per-example result schema shared by every benchmark oracle.

Before this module each oracle wrote its own column list, and they had drifted:
different names for the same quantity, different orders, and columns present in
one file and missing from another. Anything joining across benchmarks had to
special-case each one.

The schema exists to support two analyses:

Paired comparison
    `benchmark_item_id` is stable across every strategy and model, so
    Direct-vs-enhanced comparisons on the same items are a join, not a
    reconstruction. This is what makes McNemar tests and paired accuracy
    differences possible later. `run_id` and `seed` identify the run the row
    came from; `manifest_version` identifies the item selection.

Cost comparison
    input/output/total tokens and `model_calls` are summed over EVERY model
    call the invocation made, so a two-call strategy is charged for two calls.

Correctness vs. validity
    `correct` is semantic: it compares against benchmark ground truth.
    `validation_status` is structural: whether the value had the declared type.
    They are independent, and `failure_type` keeps them that way -- a
    structurally invalid output and a well-formed wrong answer are different
    events, and a run that hit a provider error is neither.

Deriving failure_type is the one piece of real logic here. The runtime sets
execution-level failures (it can see a parse failure or a type violation); only
the harness can see that a valid value disagrees with ground truth, so the
semantic category is filled in here and never overrides a runtime failure.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from aicomputing.request import ExecuteResult, FailureType

# Column order is the file format. Append new columns at the end rather than
# inserting, so existing readers keep working.
RESULT_COLUMNS: List[str] = [
    # --- identity: what was run -------------------------------------------
    "run_id",
    "timestamp",
    "model",
    "strategy",
    "is_baseline",
    # --- identity: what it was run on -------------------------------------
    "benchmark",
    "benchmark_group",
    "benchmark_item_id",
    "benchmark_category",
    "invocation_id",
    # --- the specification and the call -----------------------------------
    "signature",
    "docstring",
    "input_line",
    "inputs",
    # --- outcome ----------------------------------------------------------
    "expected",
    "predicted",
    "correct",
    "scorable",
    "validation_status",
    "failure_type",
    "parse_status",
    # --- self-checking (contract) -----------------------------------------
    "contract_checked",
    "contract_revised",
    "revisions",
    "pre_revision_value",
    "pre_revision_correct",
    # --- cost -------------------------------------------------------------
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "model_calls",
    "retries",
    "token_retries",
    # --- provenance / diagnostics -----------------------------------------
    "seed",
    "manifest_version",
    "dataset_revision",
    "invocation_pool",
    "return_type_source",
    "requested_think",
    "effective_think",
    "thinking_honoured",
    "temperature",
    "top_p",
    "max_new_tokens",
    "exec_status",
    "type_warnings",
    "raw_text",
    "prompts",
]


def _json(value: Any) -> str:
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value))


def derive_failure_type(result: ExecuteResult, correct: bool) -> Optional[str]:
    """
    Classify the outcome, preferring the runtime's execution-level verdict.

    A runtime failure always wins over the semantic one. If the output could
    not be parsed, or was withheld by STRICT validation, then the model never
    produced an answer to be wrong about, and labelling it a wrong answer
    would inflate the apparent semantic error rate with what are really
    execution failures.
    """
    if result.failure_type:
        return result.failure_type

    # Code-generation baseline outcomes. These are the model's program
    # failing, not the provider failing, so they map to GENERATED_CODE_*
    # (scorable) and never to provider_error. Reporting a program that does
    # not run as an API failure would remove it from the denominator and
    # flatter the baseline exactly where it is weakest.
    codegen = {
        "syntax_error": FailureType.GENERATED_CODE_SYNTAX,
        "runtime_error": FailureType.GENERATED_CODE_RUNTIME,
        "timeout": FailureType.GENERATED_CODE_TIMEOUT,
        # Legacy status from runs predating the split; it covered both
        # syntax and runtime failures, so the less specific one is used.
        "error": FailureType.GENERATED_CODE_RUNTIME,
    }
    if result.exec_status in codegen:
        return codegen[result.exec_status]

    if not correct:
        return FailureType.SEMANTIC_WRONG_ANSWER
    return FailureType.SUCCESS


def build_row(
    result: ExecuteResult,
    *,
    run_id: str,
    benchmark: str,
    benchmark_group: str,
    item_id: str,
    signature: str,
    docstring: str,
    inputs: Dict[str, Any],
    expected: Any,
    correct: bool,
    seed: int,
    manifest_version: Any,
    category: Optional[str] = None,
    invocation_id: Optional[str] = None,
    input_line: str = "",
    is_baseline: bool = False,
    pre_revision_correct: Optional[bool] = None,
    timestamp: Optional[str] = None,
    dataset_revision: Optional[str] = None,
    invocation_pool: Optional[str] = None,
    return_type_source: Optional[str] = None,
    generation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble one schema-conformant result row."""
    checked = result.revised is not None      # a check ran at all
    failure = derive_failure_type(result, correct) or ""
    gen = generation or {}
    return {
        "run_id": run_id,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": result.model,
        "strategy": result.strategy,
        "is_baseline": is_baseline,
        "benchmark": benchmark,
        "benchmark_group": benchmark_group,
        "benchmark_item_id": item_id,
        "benchmark_category": category or "",
        "invocation_id": invocation_id or "",
        "signature": signature,
        "docstring": docstring,
        "input_line": input_line,
        "inputs": _json(inputs),
        "expected": _json(expected),
        "predicted": _json(result.result),
        "correct": correct,
        # Whether this invocation belongs in the capability denominator.
        # A provider outage is not the model being wrong.
        "scorable": (result.scorable
                     and FailureType.is_scorable(failure or None)),
        "validation_status": result.validation_status,
        "failure_type": failure,
        "parse_status": result.parse_status or "",
        "contract_checked": checked,
        "contract_revised": result.revised if checked else "",
        "revisions": int(bool(result.revised)) if checked else 0,
        "pre_revision_value": _json(result.pre_revision_value) if checked else "",
        "pre_revision_correct": "" if pre_revision_correct is None else pre_revision_correct,
        # Latency is always a real number; None would crash the writer and
        # lose the file at the end of a cell (see bench_args.execute_resilient).
        "latency_ms": round(result.latency_ms or 0.0),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.total_tokens,
        "model_calls": result.model_calls,
        "retries": result.retries,
        "token_retries": result.token_retries,
        "seed": seed,
        "manifest_version": manifest_version,
        "dataset_revision": dataset_revision or "",
        "invocation_pool": invocation_pool or "",
        "return_type_source": return_type_source or "",
        # Thinking actually used, not what the strategy asked for.
        "requested_think": result.requested_think,
        "effective_think": result.effective_think,
        "thinking_honoured": result.thinking_honoured,
        # Resolved generation settings, recorded so a row states the
        # configuration it was produced under rather than requiring the
        # manifest to be consulted separately.
        "temperature": gen.get("temperature"),
        "top_p": gen.get("top_p"),
        "max_new_tokens": gen.get("max_new_tokens"),
        "exec_status": result.exec_status or "",
        "type_warnings": "; ".join(result.type_warnings),
        "raw_text": result.raw_text,
        "prompts": _json(result.prompts),
    }


def write_results(path: str, rows: Iterable[Dict[str, Any]]) -> str:
    """Write rows as CSV in the canonical column order."""
    rows = list(rows)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in RESULT_COLUMNS})
    return path


def new_run_id(model: str, benchmark: str, strategy: str) -> str:
    """A short, sortable, human-readable identifier for one cell."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = lambda s: "".join(c if c.isalnum() or c in "-._" else "-" for c in s)
    return f"{stamp}_{safe(model)}_{safe(benchmark)}_{safe(strategy)}"
