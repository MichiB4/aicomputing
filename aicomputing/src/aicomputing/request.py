"""
Request, result, and failure data structures for AI Computing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .validation import Constraint, ValidationMode


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
    """
    A request to execute an AI function.

    The programmer declares what the function does through its signature
    and docstring, provides concrete inputs, and the VirtualRuntime
    predicts the output.
    """

    signature: str
    docstring: str
    inputs: Dict[str, Any]

    # Expected output type; None uses the signature's return annotation.
    return_type: Optional[Any] = None

    # Optional constraint predicates on the output.
    constraints: List[Constraint] = field(default_factory=list)

    # Generation parameters.
    temperature: float = 0.0
    top_p: float = 1.0

    # Positive initial output-token budget, or None for no application cap.
    # Providers still enforce their own limits. Token-limit retries may grow
    # an explicit budget; validation retries are configured separately below.
    max_new_tokens: Optional[int] = None

    # Strategy override; None uses the runtime default.
    strategy: Optional[str] = None

    # Model override; None uses the runtime default.
    model: Optional[str] = None

    # Controls input type checking: True raises on mismatch, False records
    # warnings, and None inherits the runtime's strict_types setting.
    # Also selects STRICT or WARN output validation when validation_mode
    # is not explicitly set on this request.
    strict_types: Optional[bool] = None

    # Output validation policy; overrides this request's strict_types setting.
    # If both are None, inherits the runtime's validation_mode.
    # STRICT withholds invalid output; WARN returns it with warnings.
    validation_mode: ValidationMode | str | bool | None = None

    # Non-negative maximum additional executions after parsing or validation
    # failure, separate from token-budget retries within a strategy.
    max_retries: int = 1


    def __post_init__(self) -> None:
        self.validate_limits()

    def validate_limits(self) -> None:
        """Check generation limits at construction and again before execution."""
        validate_generation_limits(self.max_new_tokens, self.max_retries)


class FailureType:
    """
    Failure identifiers for execution and benchmark evaluation.

    Distinguishes valid but incorrect answers, outputs that could not be
    parsed or validated, generated-code failures, and infrastructure failures.
    Infrastructure failures are reported separately from model capability.

    Execution failures are recorded by the runtime and benchmark wrapper.
    The benchmark harness determines semantic correctness against ground truth.
    """

    # No failure recorded; does not establish correctness.
    SUCCESS = None

    # Output failures: incorrect, invalid, or unparseable results.

    # Structurally valid output that fails benchmark comparison.
    SEMANTIC_WRONG_ANSWER = "semantic_wrong_answer"
    STRUCTURAL_VALIDATION = "structural_validation_failure"
    CONSTRAINT_VALIDATION = "constraint_validation_failure"
    PARSE = "parse_failure"  # Parsing failed to recover an output value.

    # Generated-code failures, used only by the code-generation baseline.
    # Distinguish failures of generated programs from provider/API failures.
    GENERATED_CODE_TRUNCATED = "generated_code_truncated"
    GENERATED_CODE_SYNTAX = "generated_code_syntax_error"
    GENERATED_CODE_RUNTIME = "generated_code_runtime_error"
    GENERATED_CODE_TIMEOUT = "generated_code_timeout"
    GENERATED_CODE_INVALID_OUTPUT = "generated_code_invalid_output"

    # Infrastructure failures: provider communication or execution harness.
    PROVIDER_ERROR = "provider_error"  # API/network/auth failure.
    TIMEOUT = "timeout"  # Inference request timed out.
    HARNESS_ERROR = "harness_error"  # Runtime or evaluation harness failure.

    # Failure categories included in the capability-accuracy denominator.
    # Infrastructure failures are excluded and reported separately.
    SCORABLE = frozenset({
        SEMANTIC_WRONG_ANSWER,
        STRUCTURAL_VALIDATION,
        CONSTRAINT_VALIDATION,
        PARSE,
        GENERATED_CODE_TRUNCATED,
        GENERATED_CODE_SYNTAX,
        GENERATED_CODE_RUNTIME,
        GENERATED_CODE_TIMEOUT,
        GENERATED_CODE_INVALID_OUTPUT,
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
        """
        Return whether the outcome belongs in capability-accuracy scoring.

        None and an empty string represent no recorded failure. Listed output
        and generated-code failures are included; infrastructure failures are
        excluded. Unknown labels raise ValueError; invalid types raise TypeError.
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

    # Revision-check outcome: None = no check recorded, False = original retained,
    # True = revision reported. This does not establish correctness.
    revised: Optional[bool] = None

    # Prediction before the revision check; meaningful when revised is not None.
    pre_revision_value: Any = None

    # Provider-reported finish reason before normalization, when available.
    provider_finish_reason: Optional[str] = None

    # Prompt texts recorded by the strategy on the final execution attempt.
    # Entries do not necessarily correspond one-to-one with model calls.
    prompts: List[str] = field(default_factory=list)

    # Code-generation/execution status, including truncated generation before execution.
    # None for predictive strategies.
    exec_status: Optional[str] = None

    # Whether this invocation returned a cached prediction.
    cached: bool = False

    # Input validation warnings; strict input checking raises instead.
    input_warnings: List[str] = field(default_factory=list)

    # Output type warnings from the final attempt, including STRICT failures.
    type_warnings: List[str] = field(default_factory=list)

    # Output constraint warnings from the final attempt, including STRICT failures.
    constraint_warnings: List[str] = field(default_factory=list)

    # Outcome of applicable output checks; "valid" does not establish correctness.
    # Values: "valid", "invalid_type", "invalid_constraint", "unparsed".
    validation_status: str = "valid"

    # Evaluation metadata: eligibility for the capability-accuracy denominator.
    # True does not mean correct.
    scorable: bool = True

    # Parsing outcome: "parsed", "fallback_parsed", or "failed".
    # Fallback parsing is not necessarily a failure for string outputs.
    # None means no parsing status was recorded.
    parse_status: Optional[str] = None

    # Parsing diagnostic, when available.
    parse_error: Optional[str] = None

    # FailureType identifier, or None when no failure is recorded.
    # Absence of a failure does not establish prediction correctness.
    # Benchmark scoring derives semantic wrong-answer classification separately.
    failure_type: Optional[str] = None

    # Diagnostics.

    # Additional strategy executions initiated by output-validation retries.
    retries: int = 0

    # Token-budget retries reported by the final strategy execution attempt.
    token_retries: int = 0

    # Legacy token count reported by the final strategy attempt.
    # Retained pending consumer review; prefer accumulated output_tokens.
    tokens_used: Optional[int] = None

    # Token usage accumulated across recorded model responses, including retries.
    # None represents unavailable usage; completeness depends on provider reporting.
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Model-call count recorded for the invocation, including retries.
    model_calls: int = 0

    # Native reasoning overrides for each response recorded in this invocation.
    # Entries preserve call order: {} means no override; None means unrecorded.
    # Cache hits have no new calls and therefore an empty list.
    reasoning_requests: List[Optional[Dict[str, Any]]] = field(default_factory=list)

    # Output-token ceiling reported by the strategy, not tokens consumed.
    # None means no application cap, or no budget recorded (e.g. a cache hit).
    token_budget_used: Optional[int] = None

    # Elapsed invocation time in milliseconds, including execution retries.
    # Runtime measurements exclude subsequent result processing and trace writing.
    latency_ms: Optional[float] = None

    # Ordered observations per recorded response, separate from requested settings.
    # None entries mean unrecorded observations; cache hits have no entries.
    # These measurements do not establish the model's internal reasoning mode.
    reasoning_observations: List[Optional[Dict[str, Any]]] = field(default_factory=list)
