"""Strategy interface, shared prediction calls, and response accounting.

Prediction strategies return model-produced values; code_execute runs a
generated implementation. Native reasoning follows backend configuration.

None leaves output uncapped by the application. Explicit token limits stay
fixed on every call. Validation retries are controlled by the runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from ..backends.base import Backend, GenerateResponse
from ..parsing import ParseOutcome, coerce_type, parse_result
from ..prompting import build_retry_prompt
from ..request import ExecuteRequest
from ..status import ParseStatus
from ..validation import ValidationMode, validate_bool

@dataclass
class CallStats:
    """Accumulate explicitly recorded responses, including all stages and retries.

    Token counts remain None until reported; zero means a reported zero.
    """

    model_calls: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    # Response-order snapshots: None = unrecorded, {} = omitted; not actual mode.
    reasoning_requests: List[Optional[dict[str, Any]]] = field(default_factory=list)

    # False once any recorded response omits the corresponding token count.
    # The counts above can therefore contain partial sums.
    input_complete: bool = True
    output_complete: bool = True

    # One snapshot per returned response; None means no observations recorded.
    reasoning_observations: List[Optional[dict[str, Any]]] = field(default_factory=list)
    transport_diagnostics: List[Optional[dict[str, Any]]] = field(default_factory=list)

    def record(self, response: "GenerateResponse") -> "CallStats":
        """Record usage, native requests, and response observations separately."""
        self.model_calls += 1

        self.reasoning_requests.append(deepcopy(response.requested_reasoning))
        self.reasoning_observations.append(deepcopy(response.observed_reasoning))
        self.transport_diagnostics.append(deepcopy(response.transport_diagnostics))
        raw_in = response.input_tokens
        raw_out = response.output_tokens
        if raw_in is None:
            self.input_complete = False
        if raw_out is None:
            self.output_complete = False

        self.input_tokens = _add_optional(self.input_tokens, raw_in)
        self.output_tokens = _add_optional(self.output_tokens, raw_out)
        return self

    @property
    def total_tokens(self) -> Optional[int]:
        """Return total usage, or None if either component is missing/incomplete."""
        if not (self.input_complete and self.output_complete):
            return None
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


def _add_optional(a: Optional[int], b: Optional[int]) -> Optional[int]:
    """Add reported counts, retaining a partial sum when one is unknown."""
    if a is None:
        return b
    if b is None:
        return a
    return a + b


@dataclass
class StrategyResult:
    """Result from a strategy execution (before runtime validation)."""
    value: Any
    # Strategy response text; composite strategies may combine or annotate replies.
    raw_text: str

    # Stage-order prompts; repeated samples are not separate entries.
    prompts: List[str] = field(default_factory=list)

    # Preserve the parsing view for retry feedback separately from raw evidence.
    feedback_text: Optional[str] = None

    # Strategy selected by hybrid, when applicable.
    substrategy: Optional[str] = None

    # Original reason for the selected response, not a complete call history.
    provider_finish_reason: Optional[str] = None

    # Accumulated model-call accounting, when supplied by the strategy.
    # None makes invocation token totals unknown.
    call_stats: Optional[CallStats] = None

    # Parsing status and diagnostic; explicit FAILED makes an attached value unavailable.
    parse_status: Optional[ParseStatus] = None
    parse_error: Optional[str] = None

    # Code extraction/execution diagnostics are separate from prediction parsing.
    execution_error: Optional[str] = None

    # Contract checker verdict parsing, separate from answer parsing.
    verdict_parse_status: Optional[ParseStatus] = None
    verdict_parse_error: Optional[str] = None
    verdict_finish_reason: Optional[str] = None

    # Whether revision changed the candidate; None if no check was reported.
    revised: Optional[bool] = None

    # Original candidate before revision, when applicable.
    pre_revision_value: Any = None

    # Generated-source strategies retain source and execution diagnostics.
    generated_code: Optional[str] = None
    execution_details: Optional[dict[str, Any]] = None

    # A strategy may use its own response protocol instead of value framing.
    answer_format: Optional[str] = None

    def __post_init__(self) -> None:
        if self.parse_status is not None:
            self.parse_status = ParseStatus(self.parse_status)
        if self.verdict_parse_status is not None:
            self.verdict_parse_status = ParseStatus(self.verdict_parse_status)


class Strategy(ABC):
    """Abstract base for execution strategies."""

    supports_validation_retries = True

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'direct', 'cot', 'code_execute'."""

    def cache_state(self) -> dict[str, Any]:
        """Describe cache configuration; unsupported values bypass caching.

        Represent delegates by their configuration rather than strategy objects.
        """
        return vars(self).copy()

    @abstractmethod
    def execute(
        self,
        request: ExecuteRequest,
        backend: Backend,
        feedback: Optional[dict] = None,
    ) -> StrategyResult:
        """Produce a value with the supplied backend.

        Retry feedback carries the previous response and validation message
        under "raw_output" and "error".
        """


class PromptStrategy(Strategy):
    """Predict with one prompt from ``prompt_builder``; subclasses set ``name``."""

    prompt_builder: Callable[..., str]

    def __init__(self, include_fewshot: bool = True) -> None:
        validate_bool(include_fewshot, "include_fewshot")
        self._include_fewshot = include_fewshot

    def execute(
        self,
        request: ExecuteRequest,
        backend: Backend,
        feedback: Optional[dict] = None,
    ) -> StrategyResult:
        prompt = self.prompt_builder(
            signature=request.signature,
            docstring=request.docstring,
            inputs=request.inputs,
            include_fewshot=self._include_fewshot,
        )
        return _execute_prompt(request, backend, prompt, feedback)


def _with_feedback(prompt: str, feedback: Optional[dict]) -> str:
    """Append validation-retry feedback to a prompt, when supplied."""
    if not feedback:
        return prompt
    return build_retry_prompt(
        original_prompt=prompt,
        raw_output=feedback.get("raw_output", ""),
        error_message=feedback.get("error", ""),
    )


def _backend_call(
    request: ExecuteRequest, backend: Backend, prompt: str,
) -> Callable[[], GenerateResponse]:
    """Bind the request's generation settings, including its fixed token limit."""
    def call() -> GenerateResponse:
        return backend.generate(
            prompt, model=request.model, temperature=request.temperature,
            top_p=request.top_p, max_new_tokens=request.max_new_tokens,
        )
    return call


def _execute_prompt(
    request: ExecuteRequest,
    backend: Backend,
    prompt: str,
    feedback: Optional[dict] = None,
) -> StrategyResult:
    """Generate and parse one prediction prompt, preserving usage evidence."""
    prompt = _with_feedback(prompt, feedback)
    stats = CallStats()
    outcome, response = _generate_once(
        call=_backend_call(request, backend, prompt),
        parse=make_parser(_output_type(request)), stats=stats,
    )
    return StrategyResult(
        value=outcome.value, raw_text=response.text, prompts=[prompt],
        feedback_text=response.parsing_text,
        provider_finish_reason=response.provider_finish_reason,
        call_stats=stats, parse_status=outcome.status, parse_error=outcome.error,
    )


def _generate_once(
    call: Callable[[], GenerateResponse],
    parse: Callable[[str], ParseOutcome],
    stats: CallStats,
) -> tuple[ParseOutcome, GenerateResponse]:
    """Record one response and never decode a provider-reported truncated reply."""
    response = call()
    stats.record(response)
    if response.finish_reason == "length":
        outcome = ParseOutcome(None, ParseStatus.FAILED, "response truncated at output limit")
    else:
        outcome = parse(response.parsing_text)
    return outcome, response


def _same_value(a: Any, b: Any) -> bool:
    """Compare exact types and contents, ignoring dictionary insertion order.

    Sequence order matters; set order does not. Scalar values use their
    normal equality operation, without string or repr fallbacks.
    """
    if type(a) is not type(b):
        return False

    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(
            _same_value(left, right) for left, right in zip(a, b)
        )

    if isinstance(a, dict):
        if len(a) != len(b):
            return False
        for left_key, left_value in a.items():
            if not any(
                _same_value(left_key, right_key)
                and _same_value(left_value, right_value)
                for right_key, right_value in b.items()
            ):
                return False
        return True

    if isinstance(a, (set, frozenset)):
        if len(a) != len(b):
            return False
        for left in a:
            if not any(_same_value(left, right) for right in b):
                return False
        return True

    return bool(a == b)


def _accumulate(total: "CallStats", part: "Optional[CallStats]") -> "CallStats":
    """Merge an attempt's reported accounting into total, in place."""
    if part is None:
        # Missing attempt accounting makes invocation token totals incomplete.
        total.input_complete = False
        total.output_complete = False
        return total
    total.model_calls += part.model_calls
    total.input_tokens = _add_optional(total.input_tokens, part.input_tokens)
    total.output_tokens = _add_optional(total.output_tokens, part.output_tokens)
    total.reasoning_requests.extend(deepcopy(part.reasoning_requests))
    total.reasoning_observations.extend(deepcopy(part.reasoning_observations))
    total.transport_diagnostics.extend(deepcopy(part.transport_diagnostics))
    # Any incomplete component keeps the aggregate component incomplete.
    total.input_complete = total.input_complete and part.input_complete
    total.output_complete = total.output_complete and part.output_complete
    return total


def _output_type(request: ExecuteRequest) -> Any:
    """Use the signature's output type unless output checks are disabled."""
    if ValidationMode.coerce(request.validation_mode) is ValidationMode.NONE:
        return Any
    return request.return_type


def make_parser(return_type) -> Callable[[str], ParseOutcome]:
    """Decode a prediction and apply the selected output conversions."""
    def parse(text: str) -> ParseOutcome:
        outcome = parse_result(text)
        outcome.value = coerce_type(outcome.value, return_type)
        return outcome
    return parse
