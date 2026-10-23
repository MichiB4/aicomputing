"""
Canonical per-invocation CSV schema for the manifest-driven benchmark runner.

RESULT_COLUMNS defines output column order. The legacy oracle scripts retain
their separate CSV formats.

Rows record run and task identity, specifications and inputs, expected and
predicted values, validation and failure diagnostics, contract revisions,
reported usage, and generation settings.

Correctness is supplied by the benchmark grader. Structural validation does
not establish semantic correctness. Failure classification preserves explicit
execution failures before considering whether an answer matches ground truth.

Token and model-call fields reflect accounting recorded in ExecuteResult.
Unavailable usage remains unknown; failed calls may have incomplete accounting.

manifest_version records the manifest format version, not a unique experiment
or item selection. Paired analysis also requires matching task invocations,
data, inputs, and the relevant experiment settings.
"""

from __future__ import annotations

import csv
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from aicomputing.request import ExecuteResult, FailureType

# Canonical column order for newly written CSV files.
# Consumers should address columns by name; older files may use an earlier schema.
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
    "temperature",
    "top_p",
    "max_new_tokens",
    "exec_status",
    "type_warnings",
    "raw_text",
    "prompts",
    "reasoning_requests",
    "reasoning_observations",
]


def _json(value: Any) -> str:
    """Encode a CSV field as JSON, falling back to string representations."""
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value))


def derive_failure_type(result: ExecuteResult, correct: bool) -> Optional[str]:
    """Classify a benchmark outcome without replacing explicit failures.

    Validate existing failure labels first. If none is recorded, map known
    generated-code execution failures, then use grader-supplied correctness.
    Scoring eligibility is determined separately when building the row.
    """
    # Validate explicit labels even when they are falsy or already excluded.
    FailureType.is_scorable(result.failure_type)
    if result.failure_type:
        return result.failure_type

    # Generation/execution failures remain scorable model outcomes rather
    # than being excluded as provider failures.
    codegen = {
        "truncated": FailureType.GENERATED_CODE_TRUNCATED,
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
    if (
        result.exec_status in ("truncated", "syntax_error", "runtime_error", "timeout", "error")
        or failure in (
            FailureType.PARSE,
            FailureType.GENERATED_CODE_TRUNCATED,
            FailureType.GENERATED_CODE_SYNTAX,
            FailureType.GENERATED_CODE_RUNTIME,
            FailureType.GENERATED_CODE_TIMEOUT,
        )
    ):
        correct = False
    failure_scorable = FailureType.is_scorable(failure)
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
        "scorable": result.scorable and failure_scorable,
        "validation_status": result.validation_status,
        "failure_type": failure,
        "parse_status": result.parse_status or "",
        "contract_checked": checked,
        "contract_revised": result.revised if checked else "",
        "revisions": int(bool(result.revised)) if checked else 0,
        "pre_revision_value": _json(result.pre_revision_value) if checked else "",
        "pre_revision_correct": "" if pre_revision_correct is None else pre_revision_correct,
        # Record integer milliseconds; missing latency currently defaults to zero.
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
        # Native settings sent for recorded responses, not verified reasoning mode.
        "reasoning_requests": _json(result.reasoning_requests),
        "reasoning_observations": _json(result.reasoning_observations),
    }


def write_results(path: str, rows: Iterable[Dict[str, Any]]) -> str:
    """Write a complete CSV before replacing the destination.

    Unknown fields raise; missing fields are written as blanks.
    Existing permission bits are preserved; new files use temporary-file defaults.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    try:
        existing_mode = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        existing_mode = None

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=directory,
            prefix=".results-",
            suffix=".csv",
            delete=False,
        ) as fh:
            temporary_path = fh.name
            writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        if existing_mode is not None:
            os.chmod(temporary_path, existing_mode)

        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except PermissionError:
                if os.name != "nt":
                    raise
                # Windows cannot delete a read-only temporary output.
                os.chmod(temporary_path, stat.S_IWRITE)
                os.unlink(temporary_path)

    return path


def new_run_id(model: str, benchmark: str, strategy: str) -> str:
    """Combine a UTC timestamp, readable labels and a random run identifier."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def safe(value: str) -> str:
        return "".join(
            character if character.isalnum() or character in "-._" else "-"
            for character in value
        )

    return (
        f"{stamp}_{safe(model)}_{safe(benchmark)}_{safe(strategy)}"
        f"_{uuid4().hex}"
    )
