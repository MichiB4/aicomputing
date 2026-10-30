"""Invocation requests, results, and failure classifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .validation import Constraint, ValidationMode
from .answer_protocol import FRAMED_TEXT_V1, LEGACY, validate_answer_format


def validate_generation_limits(max_new_tokens: Optional[int], max_retries: int) -> None:
    """Validate starting token budget and additional validation retries."""
    if max_new_tokens is not None:
        if type(max_new_tokens) is not int:
            raise TypeError("max_new_tokens must be an integer or None")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
    if type(max_retries) is not int:
        raise TypeError("max_retries must be an integer")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")


@dataclass
class ExecuteRequest:
    """Function specification, concrete inputs, and prediction settings."""

    signature: str
    docstring: str
    inputs: Dict[str, Any]

    # Expected output type; None uses the signature's return annotation.
    return_type: Optional[Any] = None

    # Output constraint predicates.
    constraints: List[Constraint] = field(default_factory=list)

    # Generation parameters.
    temperature: float = 0.0
    top_p: float = 1.0

    # Initial token allowance; None omits the application cap. Truncation retries
    # may increase an explicit allowance, subject to provider limits.
    max_new_tokens: Optional[int] = None

    # Strategy override; None uses the runtime default.
    strategy: Optional[str] = None

    # Model override; None uses the runtime default.
    model: Optional[str] = None

    # Input mismatches: True raises, False warns, None inherits the runtime.
    # Also selects output STRICT/WARN when validation_mode is unset.
    strict_types: Optional[bool] = None

    # Output policy: overrides strict_types, otherwise inherits the runtime.
    # STRICT withholds invalid output; WARN returns it with diagnostics.
    validation_mode: ValidationMode | str | bool | None = None

    # Additional executions after parse/validation failure; excludes token retries.
    max_retries: int = 1

    # Legacy remains the default; reviewed evaluations select framed_text_v1.
    answer_format: str = LEGACY

    def __post_init__(self) -> None:
        self.validate_limits()

    def validate_limits(self) -> None:
        """Check generation limits at construction and again before execution."""
        validate_generation_limits(self.max_new_tokens, self.max_retries)
        validate_answer_format(self.answer_format)
        if self.answer_format == FRAMED_TEXT_V1 and self.max_retries != 0:
            raise ValueError("framed_text_v1 requires max_retries=0")


class FailureType:
    """Execution and benchmark failure identifiers.

    Runtime/wrappers record execution failures; evaluation determines semantic
    correctness. Infrastructure failures are excluded from capability accuracy.
    """

    # No failure recorded; does not establish correctness.
    SUCCESS = None

    # Structurally valid output that fails benchmark comparison.
    SEMANTIC_WRONG_ANSWER = "semantic_wrong_answer"
    REFERENCE_TYPE_MISMATCH = "reference_type_mismatch"  # Evaluation only.
    STRUCTURAL_VALIDATION = "structural_validation_failure"
    CONSTRAINT_VALIDATION = "constraint_validation_failure"
    PARSE = "parse_failure"  # Parsing failed to recover an output value.

    # Infrastructure failures: provider communication or execution harness.
    PROVIDER_ERROR = "provider_error"  # API/network/auth failure.
    TIMEOUT = "timeout"  # Inference request timed out.
    HARNESS_ERROR = "harness_error"  # Runtime or evaluation harness failure.

    # Capability accuracy includes output failures; infrastructure is separate.
    SCORABLE = frozenset({
        SEMANTIC_WRONG_ANSWER,
        REFERENCE_TYPE_MISMATCH,
        STRUCTURAL_VALIDATION,
        CONSTRAINT_VALIDATION,
        PARSE,
    })

    INFRASTRUCTURE = frozenset({
        PROVIDER_ERROR,
        TIMEOUT,
        HARNESS_ERROR,
    })

    # All failure identifiers; success is represented separately by None.
    ALL = tuple(sorted(SCORABLE | INFRASTRUCTURE))

    @classmethod
    def is_scorable(cls, failure_type: str | None) -> bool:
        """Include successes and output failures in capability accuracy.

        None and "" mean no recorded failure. Infrastructure is excluded.
        Unknown labels raise ValueError; invalid types raise TypeError.
        """
        if failure_type is None:
            return True
        if not isinstance(failure_type, str):
            raise TypeError("failure_type must be a string or None")
        if failure_type == "" or failure_type in cls.SCORABLE:
            return True
        if failure_type in cls.INFRASTRUCTURE:
            return False
        raise ValueError(f"Unknown failure type: {failure_type!r}")


@dataclass
class ExecuteResult:
    """Predicted output and diagnostics from an AI function invocation."""

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

    # Provider-reported finish reason before normalization, when available.
    provider_finish_reason: Optional[str] = None

    # Final-attempt prompts; entries need not correspond one-to-one with calls.
    prompts: List[str] = field(default_factory=list)

    # Whether this invocation returned a cached prediction.
    cached: bool = False

    # Input validation warnings; strict input checking raises instead.
    input_warnings: List[str] = field(default_factory=list)

    # Output type warnings from the final attempt, including STRICT failures.
    type_warnings: List[str] = field(default_factory=list)

    # Output constraint warnings from the final attempt, including STRICT failures.
    constraint_warnings: List[str] = field(default_factory=list)

    # "valid", "invalid_type", "invalid_constraint", or "unparsed"; not correctness.
    validation_status: str = "valid"

    # Included in capability-accuracy scoring; True does not mean correct.
    scorable: bool = True

    # "parsed", "fallback_parsed", "failed", or None if unrecorded.
    # Fallback strings can be valid outputs.
    parse_status: Optional[str] = None

    # Parsing diagnostic, when available.
    parse_error: Optional[str] = None

    # FailureType or None; evaluation determines semantic correctness separately.
    failure_type: Optional[str] = None

    # Additional strategy executions initiated by output-validation retries.
    retries: int = 0

    # Token-budget retries reported by the final strategy execution attempt.
    token_retries: int = 0

    # Legacy final-attempt token count; prefer accumulated output_tokens.
    tokens_used: Optional[int] = None

    # Token usage accumulated across recorded model responses, including retries.
    # None represents unavailable usage; completeness depends on provider reporting.
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Model-call count recorded for the invocation, including retries.
    model_calls: int = 0

    # Overrides in response order: {} = omitted, None = unrecorded; [] on cache hits.
    reasoning_requests: List[Optional[Dict[str, Any]]] = field(default_factory=list)

    # Output-token ceiling reported by the strategy, not tokens consumed.
    # None means no application cap, or no budget recorded (e.g. a cache hit).
    token_budget_used: Optional[int] = None

    # Invocation time including retries, excluding result processing and tracing.
    latency_ms: Optional[float] = None

    # Per-response observations, not inferred reasoning modes; None = unrecorded.
    # Cache hits have no entries.
    reasoning_observations: List[Optional[Dict[str, Any]]] = field(default_factory=list)

    # Versioned prompt/parser contract used for this prediction.
    answer_format: str = LEGACY

    # Per-response transport evidence; exceptions carry their own. [] on cache hits.
    transport_diagnostics: List[Optional[Dict[str, Any]]] = field(default_factory=list)
