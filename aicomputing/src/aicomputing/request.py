"""
Core request/result data structures for AI Computing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .validation import Constraint


@dataclass
class ExecuteRequest:
    """
    A request to execute an AI function.

    The programmer declares *what* the function does (signature + docstring),
    provides concrete inputs, and the VirtualRuntime predicts the output.
    """
    signature: str
    docstring: str
    inputs: Dict[str, Any]

    # Type enforcement (mirrors the formal T-AI rule)
    return_type: Optional[Any] = None

    # Optional constraint predicates on the output
    constraints: List[Constraint] = field(default_factory=list)

    # Generation parameters
    temperature: float = 0.0
    top_p: float = 1.0

    # Token budget override. None = auto (model_config.get_token_budget
    # picks 512 for non-thinking, 8192 for thinking models). Set explicitly
    # to force a specific starting budget.
    max_new_tokens: Optional[int] = None

    # Strategy override (None = use runtime default)
    strategy: Optional[str] = None

    # Model override (None = use runtime default)
    model: Optional[str] = None

    # Whether to enforce type-checking strictly (raise on mismatch).
    # None = use the runtime's strict_types setting.
    #
    # Superseded by validation_mode below, and kept because it is part of the
    # published API and is what the CLI still passes. When both are given,
    # validation_mode wins; ValidationMode.coerce() maps the boolean.
    strict_types: Optional[bool] = None

    # Structural validation policy for this call (see validation.ValidationMode).
    # None = inherit the runtime's mode. The experiment runs STRICT; library and
    # CLI callers default to WARN.
    validation_mode: Optional[Any] = None

    # Maximum retries if output fails validation
    max_retries: int = 1

    # Control thinking mode per-call (None = let strategy decide)
    think: Optional[bool] = None


class FailureType:
    """
    Why an invocation did not produce a correct answer.

    Deliberately not a single "failed" flag. A model that returns 3 when the
    answer is 4 has executed the specification and got it wrong; a model whose
    output could not be parsed never executed it at all; a provider 429 means
    it was never asked. Collapsing these into one category would make a
    rate-limited run look like a capability result, which is the failure mode
    this taxonomy exists to prevent.

    Runtime-level values are set by VirtualRuntime. SEMANTIC_WRONG_ANSWER is
    set by the benchmark harness, because only ground truth can establish it --
    the runtime has no way to know the right answer.
    """

    SUCCESS = None                                   # correct; no failure

    # --- model-behaviour failures: the model was asked and answered badly ---
    SEMANTIC_WRONG_ANSWER = "semantic_wrong_answer"  # valid value, wrong answer
    STRUCTURAL_VALIDATION = "structural_validation_failure"
    CONSTRAINT_VALIDATION = "constraint_validation_failure"
    PARSE = "parse_failure"                          # no value recoverable

    # --- generated-code failures: baseline only -----------------------------
    # The LLM answered fine; its PROGRAM failed. Conflating these with
    # provider errors made "the API broke" and "the model wrote code that
    # does not run" the same number, which is the one distinction the
    # baseline exists to draw.
    GENERATED_CODE_SYNTAX = "generated_code_syntax_error"
    GENERATED_CODE_RUNTIME = "generated_code_runtime_error"
    GENERATED_CODE_TIMEOUT = "generated_code_timeout"
    GENERATED_CODE_INVALID_OUTPUT = "generated_code_invalid_output"

    # --- infrastructure: the model never got a fair chance to answer --------
    PROVIDER_ERROR = "provider_error"                # API/network/auth failure
    TIMEOUT = "timeout"                              # provider-side timeout
    HARNESS_ERROR = "harness_error"                  # our own bug

    # Outcomes that reflect model behaviour and therefore belong in a
    # capability denominator. Everything NOT listed here is infrastructure:
    # counting a 429 as a wrong answer turns a quota problem into a capability
    # claim, and the rate-limited arms would be exactly the expensive ones.
    SCORABLE = frozenset({
        SEMANTIC_WRONG_ANSWER, STRUCTURAL_VALIDATION, CONSTRAINT_VALIDATION,
        PARSE, GENERATED_CODE_SYNTAX, GENERATED_CODE_RUNTIME,
        GENERATED_CODE_TIMEOUT, GENERATED_CODE_INVALID_OUTPUT,
    })

    INFRASTRUCTURE = frozenset({PROVIDER_ERROR, TIMEOUT, HARNESS_ERROR})

    ALL = tuple(sorted(SCORABLE | INFRASTRUCTURE))

    @classmethod
    def is_scorable(cls, failure_type) -> bool:
        """
        True when the outcome reflects model behaviour.

        Success (failure_type None) is scorable. So is a wrong answer, a
        malformed one, and generated code that does not run -- all of those
        are things the model did. A provider outage is not.
        """
        if failure_type is None or failure_type == "":
            return True
        return failure_type in cls.SCORABLE


@dataclass
class ExecuteResult:
    """Result of an AI function execution."""
    result: Any
    raw_text: str
    model: str
    strategy: str

    # The strategy that actually produced the answer, when `strategy` routes
    # to another one. `hybrid` sets this to "direct" or "cot" per its keyword
    # classifier; strategies that do not delegate leave it None.
    #
    # Reported separately from `strategy` so that a hybrid run can be split
    # into "how the router decided" and "how the chosen strategy performed" —
    # otherwise a poor hybrid score cannot be attributed to either the router
    # misrouting or the sub-strategy underperforming.
    substrategy: Optional[str] = None

    # Self-revision reporting, set only by strategies with a revision step
    # (`contract`). None means the strategy never had one; False means it ran
    # the check and kept its first answer. Together with pre_revision_value
    # these give the revision rate and the flip rate — a revision that fixes a
    # wrong answer and one that breaks a right one are indistinguishable from
    # the final value alone, and they mean opposite things about whether
    # self-verification is worth its second call.
    revised: Optional[bool] = None
    pre_revision_value: Any = None

    # The provider's un-normalised finish_reason, kept so a stored run can be
    # audited for tool use or unusual terminations after the fact. The backend
    # raises ToolUseDetected on a live tool call; this is the paper trail for
    # runs already on disk.
    provider_finish_reason: Optional[str] = None

    # The prompts actually sent for this call, in order: one for the
    # single-call arms, two for `contract`, and for `consistency` the single
    # prompt it samples repeatedly. Captured by the strategy rather than
    # reconstructed from its name, so it stays correct for arms added later
    # and reflects any retry wrapping that was applied.
    prompts: List[str] = field(default_factory=list)

    # Set only by the code-generation baseline: "ok" | "error" | "timeout".
    # None for every spec-only strategy, which has no execution step. A
    # generation that does not execute scores as wrong (thesis §5.5); this
    # field keeps the failure rate reportable instead of folding it into
    # accuracy.
    exec_status: Optional[str] = None

    cached: bool = False

    # Validation info.
    #
    # input_warnings covers the *request* (a missing or wrongly typed input,
    # an argument not in the signature); type_warnings and constraint_warnings
    # cover the *result*. They are kept separate because they mean opposite
    # things during evaluation: an input warning says the harness fed the model
    # a malformed task, a result warning says the model answered badly. Merging
    # them would make a benchmark's failure attribution ambiguous.
    #
    # Only populated when validation is non-strict; strict mode raises instead.
    input_warnings: List[str] = field(default_factory=list)
    type_warnings: List[str] = field(default_factory=list)
    constraint_warnings: List[str] = field(default_factory=list)

    # How the runtime judged the final value, independent of whether it is the
    # RIGHT answer. "valid" means it matched the declared return type and any
    # constraints -- not that it is semantically correct, which only benchmark
    # ground truth can decide.
    #   "valid" | "invalid_type" | "invalid_constraint" | "unparsed"
    validation_status: str = "valid"

    # Whether this invocation belongs in a capability denominator. False for
    # infrastructure failures, which are reported separately rather than
    # silently depressing accuracy. See FailureType.is_scorable.
    scorable: bool = True

    # How the value was recovered from the model's text: "parsed",
    # "fallback_parsed" (no structured form found; raw text returned) or
    # "failed". Metadata, not a verdict -- a fallback string answer to a
    # `-> str` specification is perfectly normal.
    parse_status: Optional[str] = None
    parse_error: Optional[str] = None

    # Why this invocation failed, when it did. None on success. Set by the
    # runtime for execution-level failures; benchmark harnesses additionally
    # set FailureType.SEMANTIC_WRONG_ANSWER for a structurally valid value that
    # disagrees with ground truth. Keeping the two apart is the point: an
    # invalid object and a wrong answer are different events and pooling them
    # would hide whichever is actually driving a low score.
    failure_type: Optional[str] = None

    # Diagnostics
    retries: int = 0          # validation retries (output failed parsing/validation)
    token_retries: int = 0    # token-budget doublings (model hit length limit)

    # Completion tokens, kept as the historical name. Prefer output_tokens.
    tokens_used: Optional[int] = None

    # Full cost accounting, summed across EVERY model call the strategy made --
    # both of contract's stages, all of consistency's samples, and any retry.
    # Reporting only the final call would understate the multi-call arms by
    # exactly the amount that makes them expensive, which is the comparison
    # the cost analysis exists to make. None where a provider does not report
    # the figure; never a fabricated value.
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Number of model calls this invocation cost: 1 for the single-call arms,
    # 2 for contract, n_samples for consistency, plus any budget-doubling
    # retries. The denominator for cost-per-correct-answer.
    model_calls: int = 0

    # Thinking actually used for this invocation, resolved by the model
    # configuration layer rather than assumed from the strategy. `direct` is
    # the minimal execution condition and requests think=False; on a model
    # that cannot disable reasoning that request is not honoured, and this is
    # where that shows up. thinking_honoured=False means the strategy's
    # intended condition did not hold for this call.
    requested_think: Optional[bool] = None
    effective_think: Optional[bool] = None
    thinking_honoured: Optional[bool] = None

    # The max_new_tokens ceiling actually requested from the backend on the
    # final attempt, after any budget doubling. Not derivable from
    # token_retries alone: the starting budget varies with the model's
    # thinking mode (512 vs 8192) and can be overridden per request.
    token_budget_used: Optional[int] = None

    latency_ms: Optional[float] = None
