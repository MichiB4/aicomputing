"""Canonical rows for manifest-driven evaluations.

Graders supply correctness. Structural checks do not establish semantics, and
explicit failures take precedence over answer comparisons.
Unavailable token usage remains unknown.

Frozen manifests retain invocation/provenance details. Paired analysis requires
matching inputs and settings. retries combines validation and audited timeouts;
token-budget retry detail is omitted. Historical files keep their original schema."""

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
from aicomputing.answer_protocol import FRAMED_TEXT_V1
from aicomputing.parsing import ParseOutcome
from aicomputing.types import parse_signature
from aicomputing.validation import validate_output

# Canonical column order for newly written CSV files.
# Consumers should address columns by name; older files may use an earlier schema.
RESULT_COLUMNS: List[str] = [
    "run_id",
    "model",
    "strategy",
    "benchmark",
    "benchmark_item_id",
    "signature",
    "docstring",
    "inputs",
    "expected",
    "predicted",
    "correct",
    "scorable",
    "validation_status",
    "structural_validity",
    "failure_type",
    "parse_status",
    "parse_error",
    "contract_status",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "retries",
    "invocation_pool",
    "return_type_source",
    "temperature",
    "top_p",
    "max_new_tokens",
    "type_warnings",
    "raw_text",
    "prompts",
    "reasoning_requests",
    "reasoning_observations",
    "answer_format",
    "transport_diagnostics",
]


def _json(value: Any) -> str:
    """Encode a CSV field as JSON, falling back to string representations."""
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value))


def derive_failure_type(result: ExecuteResult, correct: bool) -> Optional[str]:
    """Classify a benchmark outcome without replacing explicit failures.

    Validate existing failure labels first, then use grader-supplied correctness.
    Scoring eligibility is determined separately when building the row.
    """
    # Validate explicit labels even when they are falsy or already excluded.
    FailureType.is_scorable(result.failure_type)
    if result.failure_type:
        return result.failure_type

    if not correct:
        return FailureType.SEMANTIC_WRONG_ANSWER
    return FailureType.SUCCESS


def _reference_types_compatible(predicted, expected):
    """Conservative top-level diagnostic; never changes grader correctness.

    Numeric and collection representations may be equivalent under a frozen
    grader. Nested value/type errors remain semantic wrong answers.
    """
    for family in ((int, float), (list, tuple, set, frozenset)):
        if type(predicted) in family and type(expected) in family:
            return True
    return type(predicted) is type(expected)


def _structural_validity(result, signature):
    """Check only a public return annotation, without converting the answer."""
    if (
        result.parse_status not in (ParseOutcome.PARSED, ParseOutcome.FALLBACK)
        or result.failure_type == FailureType.PARSE
        or result.failure_type in FailureType.INFRASTRUCTURE
        or result.validation_status == "unparsed"
    ):
        return "not_assessed"
    try:
        declared = parse_signature(signature)
    except ValueError:
        return "not_assessed"
    if declared.return_type is Any or declared.return_type is None:
        return "not_assessed"
    warnings = validate_output(result.result, declared.return_type, strict=False)
    return "invalid" if warnings else "valid"


def build_row(
    result: ExecuteResult,
    *,
    run_id: str,
    benchmark: str,
    item_id: str,
    signature: str,
    docstring: str,
    inputs: Dict[str, Any],
    expected: Any,
    correct: bool,
    invocation_pool: Optional[str] = None,
    return_type_source: Optional[str] = None,
    generation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble one schema-conformant result row."""
    contract_status = (
        "not_checked" if result.revised is None
        else "revised" if result.revised else "unchanged"
    )
    failure = derive_failure_type(result, correct) or ""
    structural_status = (
        _structural_validity(result, signature)
        if result.answer_format == FRAMED_TEXT_V1
        else "not_assessed"
    )
    if (
        result.answer_format == FRAMED_TEXT_V1
        and failure == FailureType.SEMANTIC_WRONG_ANSWER
    ):
        if structural_status == "invalid":
            failure = FailureType.STRUCTURAL_VALIDATION
        elif not _reference_types_compatible(result.result, expected):
            failure = FailureType.REFERENCE_TYPE_MISMATCH
    if failure == FailureType.PARSE:
        correct = False
    failure_scorable = FailureType.is_scorable(failure)
    gen = generation or {}
    return {
        "run_id": run_id,
        "model": result.model,
        "strategy": result.strategy,
        "benchmark": benchmark,
        "benchmark_item_id": item_id,
        "signature": signature,
        "docstring": docstring,
        "inputs": _json(inputs),
        "expected": _json(expected),
        "predicted": _json(result.result),
        "correct": correct,
        # Infrastructure failures do not enter capability accuracy.
        "scorable": result.scorable and failure_scorable,
        "validation_status": result.validation_status,
        "structural_validity": structural_status,
        "failure_type": failure,
        "parse_status": result.parse_status or "",
        "parse_error": result.parse_error or "",
        "contract_status": contract_status,
        # Record integer milliseconds; missing latency currently defaults to zero.
        "latency_ms": round(result.latency_ms or 0.0),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.total_tokens,
        "retries": result.retries,
        "invocation_pool": invocation_pool or "",
        "return_type_source": return_type_source or "",
        # Resolved generation settings for this row.
        "temperature": gen.get("temperature"),
        "top_p": gen.get("top_p"),
        "max_new_tokens": gen.get("max_new_tokens"),
        "type_warnings": "; ".join(result.type_warnings),
        "raw_text": result.raw_text,
        "prompts": _json(result.prompts),
        # Native settings sent for recorded responses, not verified reasoning mode.
        "reasoning_requests": _json(result.reasoning_requests),
        "reasoning_observations": _json(result.reasoning_observations),
        "answer_format": result.answer_format,
        "transport_diagnostics": _json(result.transport_diagnostics),
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
