"""Invocation requests, results, and failure classifications."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional

from .validation import Constraint, ValidationMode, validate_constraint_definitions, validate_nonblank_text
from .answer_protocol import ANSWER_FORMAT
from .types import parse_signature
from .status import ParseStatus, ValidationStatus


def validate_generation_limits(max_new_tokens: Optional[int], max_retries: int, *,
                               temperature: float = 0.0, top_p: float = 1.0) -> None:
    """Validate output/retry limits and provider-independent sampling bounds."""
    if max_new_tokens is not None:
        if type(max_new_tokens) is not int:
            raise TypeError("max_new_tokens must be an integer or None")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
    if type(max_retries) is not int:
        raise TypeError("max_retries must be an integer")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    for name, value in (("temperature", temperature), ("top_p", top_p)):
        if type(value) not in (int, float):
            raise TypeError(name + " must be an int or float")
        if type(value) is float and not math.isfinite(value):
            raise ValueError(name + " must be finite")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 <= top_p <= 1:
        raise ValueError("top_p must be between 0 and 1 inclusive")


@dataclass
class ExecuteRequest:
    """Function specification, concrete inputs, and execution settings."""

    signature: str
    # Nonblank behavioral specification; preserve its text literally.
    docstring: str
    inputs: Dict[str, Any]

    # Output constraint predicates.
    constraints: List[Constraint] = field(default_factory=list)

    # Finite temperature >= 0; finite top_p between 0 and 1 inclusive.
    temperature: float = 0.0
    top_p: float = 1.0

    # Fixed per-call output-token limit; None omits the application cap.
    max_new_tokens: Optional[int] = None

    # Strategy override; None uses the runtime default.
    strategy: Optional[str] = None

    # Model override; None uses the runtime default.
    model: Optional[str] = None

    # Output policy; None inherits the runtime's STRICT default.
    # WARN returns invalid values with diagnostics; NONE skips output checks.
    validation_mode: ValidationMode | str | None = None

    # Additional executions after parse/validation failure, using the same limits.
    max_retries: int = 1

    def __post_init__(self) -> None:
        self.validate_settings()

    @property
    def return_type(self) -> Any:
        """Read the output type from the signature; it cannot be overridden."""
        return parse_signature(self.signature).return_type

    def validate_settings(self) -> None:
        """Check the description, settings, and definitions without running predicates."""
        validate_nonblank_text(self.docstring, "docstring")
        validate_nonblank_text(self.model, "model", optional=True)
        validate_nonblank_text(self.strategy, "strategy", optional=True)
        validate_generation_limits(self.max_new_tokens, self.max_retries,
                                   temperature=self.temperature, top_p=self.top_p)
        if self.validation_mode is not None:
            ValidationMode.coerce(self.validation_mode)
        validate_constraint_definitions(self.constraints)


class FailureType:
    """Execution failure identifiers; absence of a failure does not prove correctness."""

    # No failure recorded; does not establish correctness.
    SUCCESS = None

    STRUCTURAL_VALIDATION = "structural_validation_failure"
    CONSTRAINT_VALIDATION = "constraint_validation_failure"
    PARSE = "parse_failure"  # Prediction-reply parsing did not recover a value.
    EXECUTION = "execution_failure"  # Generated code could not produce a value.

    # Provider communication or execution harness failures.
    PROVIDER_ERROR = "provider_error"  # API/network/auth failure.
    TIMEOUT = "timeout"  # Inference request timed out.
    HARNESS_ERROR = "harness_error"  # Local execution harness failure.


@dataclass
class ExecuteResult:
    """Returned output and diagnostics from one runtime invocation.

    Final-attempt evidence: result, raw_text, substrategy, revision fields,
    provider_finish_reason, contract-check diagnostics, prompts, warnings, statuses, error messages,
    answer_format, generated_code, and execution_details. Multi-stage strategies
    can combine response text and share prompts across calls. This is not a
    complete model-call or validation-attempt history.

    Whole-invocation evidence: retries, token counts, model_calls, latency_ms,
    reasoning_requests, reasoning_observations, and transport_diagnostics.
    Counts and response-order observations cover all recorded stages/retries.

    A cache hit retains the original final-attempt evidence. Its cached flag is
    True, inference/retry counts are zero, response-order lists are empty, and
    latency_ms measures this cache lookup. execution_details retains the earlier
    worker evidence; it does not indicate a new local execution.

    Status strings are converted to the shared enums at construction. Field
    combinations and benchmark correctness are not validated by this container.
    """

    # Returned prediction; inspect status fields to determine its availability.
    result: Any
    raw_text: str
    model: str
    strategy: str

    # Delegated strategy name, when recorded; strategy retains the top-level name.
    substrategy: Optional[str] = None

    # Revision check: None = absent, False = retained, True = revised; not correctness.
    revised: Optional[bool] = None

    # Prediction before the revision check; meaningful when revised is not None.
    pre_revision_value: Any = None

    # Selected response's provider-reported finish reason before normalization.
    provider_finish_reason: Optional[str] = None

    # Final-attempt prompts; entries need not correspond one-to-one with calls.
    prompts: List[str] = field(default_factory=list)

    # Whether this invocation returned a cached prediction.
    cached: bool = False

    # Output type warnings from the final attempt, including STRICT failures.
    type_warnings: List[str] = field(default_factory=list)

    # Output constraint warnings from the final attempt, including STRICT failures.
    constraint_warnings: List[str] = field(default_factory=list)

    # Final output checks; UNAVAILABLE is used for infrastructure outcomes.
    validation_status: ValidationStatus = ValidationStatus.NOT_CHECKED

    # Prediction parsing only; None for code_execute or unrecorded parsing.
    # Successful fallback recovery can still yield a valid output.
    parse_status: Optional[ParseStatus] = None

    # Parsing diagnostic, when available.
    parse_error: Optional[str] = None

    # Contract checker evidence, independent of the selected answer; otherwise None.
    verdict_parse_status: Optional[ParseStatus] = None
    verdict_parse_error: Optional[str] = None
    verdict_finish_reason: Optional[str] = None

    # Source-extraction/local-execution diagnostic; prediction parsing uses parse_error.
    execution_error: Optional[str] = None

    # Execution failure identifier, or None.
    failure_type: Optional[str] = None

    # Additional strategy executions initiated by output-validation retries.
    retries: int = 0

    # Token usage accumulated across recorded model responses, including retries.
    # None represents unavailable usage; completeness depends on provider reporting.
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Model-call count recorded for the invocation, including retries.
    model_calls: int = 0

    # Overrides in response order: {} = omitted, None = unrecorded; [] on cache hits.
    reasoning_requests: List[Optional[Dict[str, Any]]] = field(default_factory=list)

    # Invocation time including retries, excluding result processing and tracing.
    latency_ms: Optional[float] = None

    # Per-response observations, not inferred reasoning modes; None = unrecorded.
    # Cache hits have no entries.
    reasoning_observations: List[Optional[Dict[str, Any]]] = field(default_factory=list)

    # Versioned prompt/parser contract used for this prediction.
    answer_format: str = ANSWER_FORMAT

    # Per-response transport evidence; exceptions carry their own. [] on cache hits.
    transport_diagnostics: List[Optional[Dict[str, Any]]] = field(default_factory=list)

    # Source and local worker evidence for code_execute; None for predictions.
    generated_code: Optional[str] = None
    execution_details: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.validation_status = ValidationStatus(self.validation_status)
        if self.parse_status is not None:
            self.parse_status = ParseStatus(self.parse_status)
        if self.verdict_parse_status is not None:
            self.verdict_parse_status = ParseStatus(self.verdict_parse_status)
