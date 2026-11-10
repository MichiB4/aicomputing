"""Canonical rows for evaluation results.

Graders supply correctness. Structural checks do not establish semantics, and
explicit failures take precedence over answer comparisons.
Unavailable token usage remains unknown.

Paired analysis requires matching inputs and settings. retries combines
validation and recorded API retries.
Historical files keep their original schema."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from aicomputing.request import ExecuteResult, FailureType
from aicomputing.status import ParseStatus, ValidationStatus
from aicomputing.types import parse_signature
from aicomputing.validation import validate_output

# These labels and accuracy rules belong to evaluation, not execution results.
# policy_rejected remains recognized when reading historical worker results.
SEMANTIC_WRONG_ANSWER = "semantic_wrong_answer"
REFERENCE_TYPE_MISMATCH = "reference_type_mismatch"
_CODE_FAILURES = frozenset("code_" + status for status in (
    "format_error", "syntax_error", "policy_rejected", "runtime_error",
    "serialization_error", "timeout",
))
_OUTPUT_FAILURES = frozenset({
    SEMANTIC_WRONG_ANSWER, REFERENCE_TYPE_MISMATCH, FailureType.PARSE,
    FailureType.STRUCTURAL_VALIDATION, FailureType.CONSTRAINT_VALIDATION,
}) | _CODE_FAILURES
_INFRASTRUCTURE_FAILURES = frozenset({
    FailureType.PROVIDER_ERROR, FailureType.TIMEOUT, FailureType.HARNESS_ERROR,
})
FAILURE_TYPES = _OUTPUT_FAILURES | _INFRASTRUCTURE_FAILURES


def is_scorable(failure_type: str | None) -> bool:
    """Include successes and output failures; exclude infrastructure failures.

    Unknown labels are rejected so new failure types require an explicit policy.
    """
    if failure_type is None:
        return True
    if not isinstance(failure_type, str):
        raise TypeError("failure_type must be a string or None")
    if failure_type == "" or failure_type in _OUTPUT_FAILURES:
        return True
    if failure_type in _INFRASTRUCTURE_FAILURES:
        return False
    raise ValueError(f"Unknown failure type: {failure_type!r}")

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
    "provider_finish_reason",
    "verdict_parse_status",
    "verdict_parse_error",
    "verdict_finish_reason",
    "execution_error",
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
    """Map execution evidence to benchmark outcomes, preserving explicit failures.

    Expand the package's execution label using required worker diagnostics.
    Validate other failure labels before using grader-supplied correctness.
    Scoring eligibility is determined separately when building the row.
    """
    execution = result.execution_details
    if result.failure_type == FailureType.EXECUTION:
        if execution is None or execution["status"] == "ok":
            raise ValueError("execution_failure requires failed worker diagnostics")
    else:
        # Validate explicit labels even when they are falsy or already excluded.
        is_scorable(result.failure_type)
    if result.failure_type in _INFRASTRUCTURE_FAILURES:
        return result.failure_type
    if execution is not None and execution["status"] != "ok":
        if execution["status"] == "harness_error":
            return FailureType.HARNESS_ERROR
        failure = "code_" + execution["status"]
        is_scorable(failure)
        return failure
    if result.failure_type:
        return result.failure_type

    if not correct:
        return SEMANTIC_WRONG_ANSWER
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
    execution = result.execution_details
    recovered = (execution["status"] == "ok" if execution is not None
                 else result.parse_status in (ParseStatus.PARSED, ParseStatus.FALLBACK))
    if (
        not recovered
        or result.failure_type == FailureType.PARSE
        or result.failure_type in _INFRASTRUCTURE_FAILURES
        or result.validation_status == ValidationStatus.UNPARSED
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
    execution = result.execution_details
    # Unconverted text predictions and executed code are checked only here.
    assessed = result.validation_status == ValidationStatus.NOT_CHECKED or execution is not None
    structural_status = (
        _structural_validity(result, signature) if assessed else "not_assessed"
    )
    if assessed and failure == SEMANTIC_WRONG_ANSWER:
        if structural_status == "invalid":
            failure = FailureType.STRUCTURAL_VALIDATION
        elif not _reference_types_compatible(result.result, expected):
            failure = REFERENCE_TYPE_MISMATCH
    if failure:
        correct = False
    failure_scorable = is_scorable(failure)
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
        "scorable": failure_scorable,
        "validation_status": result.validation_status,
        "structural_validity": structural_status,
        "failure_type": failure,
        "parse_status": result.parse_status or "",
        "parse_error": result.parse_error or "",
        "provider_finish_reason": result.provider_finish_reason or "",
        "verdict_parse_status": result.verdict_parse_status or "",
        "verdict_parse_error": result.verdict_parse_error or "",
        "verdict_finish_reason": result.verdict_finish_reason or "",
        "execution_error": result.execution_error or "",
        "contract_status": contract_status,
        # Record integer milliseconds; missing latency stays unknown.
        "latency_ms": None if result.latency_ms is None else round(result.latency_ms),
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
