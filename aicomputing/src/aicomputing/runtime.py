"""Coordinate inference, validation, retries, caching, and tracing."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

from .backends import Backend, create_backend
from .answer_protocol import FRAMED_TEXT_V1
from .cache import ResultCache
from .parsing import ParseOutcome, coerce_type
from .request import ExecuteRequest, ExecuteResult, FailureType
from .strategies import Strategy, create_strategy
from .strategies.base import CallStats, StrategyResult, _accumulate
from .trace import Tracer
from .types import (
    ParsedSignature,
    UnsupportedAnnotationError,
    parse_signature,
    validate_type_annotation,
)
from .validation import (
    ValidationMode,
    AITypeError,
    validate_constraints,
    validate_inputs,
    validate_media_inputs,
    validate_output,
)

logger = logging.getLogger("aicomputing")


class VirtualRuntime:
    """
    Predict function outputs from specifications and concrete inputs.

    A backend selected by name belongs to the runtime. A supplied backend
    instance remains caller-owned and is not closed by the runtime. Use a
    context manager or close() to release owned resources. A closed runtime
    cannot execute again; create another runtime to continue.

    Example::

        with VirtualRuntime(backend="ollama", model="qwen3:4b", strategy="direct") as vr:
            result = vr.execute(ExecuteRequest(
                signature="add(x: int, y: int) -> int",
                docstring="Return the sum of x and y.",
                inputs={"x": 2, "y": 3},
                return_type=int,
            ))
        print(result.result)  # Expected answer: 5; predictions may be incorrect.
    """

    def __init__(
        self,
        backend: str | Backend = "ollama",
        model: Optional[str] = None,
        strategy: str | Strategy = "direct",
        strict_types: bool = True,
        validation_mode: ValidationMode | str | bool | None = None,
        cache_size: int = 2048,
        trace_file: Optional[str] = None,
        **backend_kwargs,
    ) -> None:
        # Named backends belong to this runtime; supplied instances are borrowed.
        self._owns_backend = isinstance(backend, str)
        self._closed = False
        if self._owns_backend:
            if model:
                backend_kwargs["model"] = model
            self._backend = create_backend(backend, **backend_kwargs)
        else:
            self._backend = backend

        try:
            # Custom strategies may depend on state that cannot be cached safely.
            self._cache_default_strategy = isinstance(strategy, str)
            self._strategy = (
                create_strategy(strategy) if self._cache_default_strategy else strategy
            )

            self._strict_types = strict_types

            # Default output-validation policy; requests can override it.
            # None selects WARN independently of the input strict_types setting.
            self._validation_mode = ValidationMode.coerce(
                validation_mode,
                ValidationMode.WARN,
            )

            self._cache = ResultCache(maxsize=cache_size)
            self._tracer = Tracer(trace_file) if trace_file else None

            self._model = model or self._backend.default_model
        except BaseException:
            if self._owns_backend:
                self._backend.close()
            raise

    def close(self) -> None:
        """Close an owned backend and clear cached predictions once.

        Supplied backend instances remain caller-owned. If backend cleanup
        raises, the runtime stays open so cleanup can be attempted again.
        Do not close while execute() is running.
        """
        if self._closed:
            return
        if self._owns_backend:
            self._backend.close()
        self._cache.clear()
        self._closed = True

    def __enter__(self) -> VirtualRuntime:
        """Enter an open runtime without loading clients or models."""
        if self._closed:
            raise RuntimeError("VirtualRuntime is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the runtime without suppressing body exceptions."""
        self.close()

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @property
    def model(self) -> str:
        return self._model

    @property
    def validation_mode(self) -> ValidationMode:
        return self._validation_mode

    @property
    def cache(self) -> ResultCache:
        return self._cache

    @property
    def tracer(self) -> Optional[Tracer]:
        return self._tracer

    def execute(self, req: ExecuteRequest) -> ExecuteResult:
        """Validate inputs, predict or reuse an output, and apply output checks.

        Failed checks may retry; eligible outputs are cached and traced.
        Strict input errors and backend exceptions may propagate.
        Predictions are not guaranteed correct.
        """
        if self._closed:
            raise RuntimeError("VirtualRuntime is closed")
        t0 = time.monotonic()
        req.validate_limits()

        # Capability restrictions apply under both input-validation policies.
        validate_media_inputs(req.inputs)

        model = req.model or self._model
        strat_name = req.strategy or self._strategy.name
        parsed_sig, input_warnings = self._validate_request_inputs(req)

        # Resolve output requirements before cache lookup to prevent bypasses.
        strategy = (
            create_strategy(strat_name)
            if strat_name != self._strategy.name
            else self._strategy
        )
        output_policy = (
            req.validation_mode
            if req.validation_mode is not None
            else req.strict_types
        )
        mode = ValidationMode.coerce(output_policy, self._validation_mode)
        output_strict = mode is ValidationMode.STRICT
        # Preserve predictions without output-type conversion or validation.
        # Benchmark reference types are used only by the evaluator.
        return_type = Any if req.answer_format == FRAMED_TEXT_V1 else req.return_type
        if return_type is None and parsed_sig is not None:
            return_type = parsed_sig.return_type

        validate_type_annotation(return_type)

        # Arbitrary predicates lack safe cache identities; run them fresh.
        named_strategy = (
            strat_name != self._strategy.name or self._cache_default_strategy
        )
        backend_state = self._backend.cache_state() if named_strategy else None
        cache_allowed = (
            named_strategy
            and backend_state is not None
            and not req.constraints
            and not input_warnings
            and parsed_sig is not None
        )
        cache_context = {
            "backend_class": (
                type(self._backend).__module__,
                type(self._backend).__qualname__,
            ),
            "backend_state": backend_state,
            "validation_mode": mode.value,
            "return_type": return_type,
            "max_retries": req.max_retries,
            "answer_format": req.answer_format,
            "strategy_class": (
                type(strategy).__module__,
                type(strategy).__qualname__,
            ),
            "strategy_state": (
                strategy.cache_state() if named_strategy else None
            ),
        }
        cached = None
        if cache_allowed:
            cached = self._cache.get(
                req.signature,
                req.docstring,
                req.inputs,
                model,
                strat_name,
                temperature=req.temperature,
                top_p=req.top_p,
                max_new_tokens=req.max_new_tokens,
                context=cache_context,
            )
        if cached is not None:
            hit = _cached_invocation(cached, input_warnings, t0)

            # Record cache hits too; they consume no new inference tokens.
            if self._tracer:
                _trace_execution(
                    self._tracer, req, hit,
                    parsed_result=hit.result, strategy=strat_name,
                )
            return hit

        retries = 0
        feedback: Optional[Dict[str, Any]] = None

        total_stats = CallStats()

        # Pass resolved settings to generation, parsing, and answer selection.
        effective_req = replace(
            req, model=model, validation_mode=mode, return_type=return_type,
        )

        for attempt in range(1 + req.max_retries):
            strat_result = strategy.execute(
                effective_req,
                self._backend,
                feedback=feedback,
            )
            _accumulate(total_stats, strat_result.call_stats)

            checked = _check_prediction(
                req, strat_result, return_type, output_strict,
            )

            # Retry failed recovery as well as type and constraint failures.
            if not (
                checked.parse_failed
                or checked.type_warnings
                or checked.constraint_warnings
            ):
                break

            if attempt < req.max_retries:
                errors = checked.type_warnings + checked.constraint_warnings
                if checked.parse_failed:
                    errors.insert(
                        0,
                        strat_result.parse_error
                        or "Could not recover an output value from the model response.",
                    )
                last_error = "; ".join(errors)
                feedback = {
                    "raw_output": strat_result.raw_text,
                    "error": last_error,
                }
                retries += 1
                logger.info(
                    "Validation retry %d/%d: %s",
                    retries,
                    req.max_retries,
                    last_error,
                )

        latency = (time.monotonic() - t0) * 1000

        # After retries, STRICT withholds invalid outputs; WARN retains them.
        validation_status = "valid"
        failure_type: Optional[str] = None
        final_value = None if checked.parse_failed else strat_result.value

        # Recovery failure takes precedence over incidental validation errors.
        # A successfully parsed None is instead checked against its output type.
        if checked.parse_failed:
            validation_status = "unparsed"
        elif checked.type_warnings:
            validation_status = "invalid_type"
        elif checked.constraint_warnings:
            validation_status = "invalid_constraint"

        if validation_status != "valid":
            failure_type = {
                "invalid_type": FailureType.STRUCTURAL_VALIDATION,
                "invalid_constraint": FailureType.CONSTRAINT_VALIDATION,
                "unparsed": FailureType.PARSE,
            }[validation_status]
            if mode is ValidationMode.STRICT and validation_status != "unparsed":
                logger.info(
                    "STRICT validation: withholding %s output (%s)",
                    validation_status,
                    "; ".join(checked.type_warnings + checked.constraint_warnings),
                )
                final_value = None

        result = ExecuteResult(
            result=final_value,
            validation_status=validation_status,
            failure_type=failure_type,
            raw_text=strat_result.raw_text,
            model=model,
            strategy=strategy.name,
            substrategy=strat_result.substrategy,
            revised=strat_result.revised,
            pre_revision_value=strat_result.pre_revision_value,
            provider_finish_reason=strat_result.provider_finish_reason,
            prompts=strat_result.prompts,
            input_warnings=input_warnings,
            type_warnings=checked.type_warnings,
            constraint_warnings=checked.constraint_warnings,
            retries=retries,
            token_retries=strat_result.retries,
            tokens_used=strat_result.tokens_used,
            token_budget_used=strat_result.token_budget_used,
            # Expose a component total only when every recorded call supplied it.
            input_tokens=(
                total_stats.input_tokens if total_stats.input_complete else None
            ),
            output_tokens=(
                total_stats.output_tokens if total_stats.output_complete else None
            ),
            total_tokens=total_stats.total_tokens,
            model_calls=total_stats.model_calls,
            parse_status=strat_result.parse_status,
            parse_error=strat_result.parse_error,
            reasoning_requests=total_stats.reasoning_requests,
            reasoning_observations=total_stats.reasoning_observations,
            answer_format=req.answer_format,
            transport_diagnostics=total_stats.transport_diagnostics,
            latency_ms=latency,
        )

        # Cache only valid predictions from eligible requests. ExecuteResult
        # distinguishes a cached None prediction from a missing cache entry.
        if cache_allowed and validation_status == "valid":
            self._cache.put(
                req.signature,
                req.docstring,
                req.inputs,
                model,
                strat_name,
                result,
                temperature=req.temperature,
                top_p=req.top_p,
                max_new_tokens=req.max_new_tokens,
                context=cache_context,
            )

        # Keep the recovered prediction in the trace even if STRICT withheld it.
        if self._tracer:
            _trace_execution(
                self._tracer, req, result,
                parsed_result=strat_result.value, strategy=strategy.name,
            )

        return result

    def _validate_request_inputs(
        self, req: ExecuteRequest,
    ) -> tuple[Optional[ParsedSignature], List[str]]:
        """Parse argument declarations and apply the effective input policy."""
        input_strict = (
            req.strict_types
            if req.strict_types is not None
            else self._strict_types
        )

        parsed_sig: Optional[ParsedSignature] = None
        input_warnings: List[str] = []
        try:
            parsed_sig = parse_signature(req.signature)
            input_warnings = validate_inputs(
                parsed_sig,
                req.inputs,
                strict=input_strict,
            )
        except UnsupportedAnnotationError:
            # Unsupported specifications are errors under either input policy.
            raise
        except (ValueError, AITypeError) as e:
            logger.warning("Input validation: %s", e)
            if input_strict:
                raise
            input_warnings = [str(e)]
        return parsed_sig, input_warnings


@dataclass
class _PredictionCheck:
    """Runtime checks for one strategy attempt, before retry selection."""

    parse_failed: bool
    type_warnings: List[str]
    constraint_warnings: List[str]


def _check_prediction(
    req: ExecuteRequest,
    strat_result: StrategyResult,
    return_type: Any,
    output_strict: bool,
) -> _PredictionCheck:
    """Coerce and check one recovered prediction, preserving failure precedence."""
    # Distinguish failed recovery from a successfully returned None.
    parse_failed = (
        strat_result.value is None
        and strat_result.parse_status not in (
            ParseOutcome.PARSED,
            ParseOutcome.FALLBACK,
        )
    )

    type_warnings: List[str] = []
    constraint_warnings: List[str] = []

    # Both policies use the same conversions and collect warnings before retries.
    if return_type is not None:
        strat_result.value = coerce_type(
            strat_result.value,
            return_type,
            strict=output_strict,
        )

        type_warnings = validate_output(
            strat_result.value,
            return_type,
            strict=False,
        )

    if req.constraints and not parse_failed and not type_warnings:
        constraint_warnings = validate_constraints(
            strat_result.value,
            req.constraints,
            strict=False,
        )
    return _PredictionCheck(
        parse_failed=parse_failed,
        type_warnings=type_warnings,
        constraint_warnings=constraint_warnings,
    )


def _cached_invocation(
    cached: ExecuteResult, input_warnings: List[str], started_at: float,
) -> ExecuteResult:
    """Reuse a prediction while recording zero new inference work."""
    return replace(
        cached,
        cached=True,
        input_warnings=input_warnings,
        retries=0,
        token_retries=0,
        tokens_used=0,
        token_budget_used=None,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        model_calls=0,
        reasoning_requests=[],
        reasoning_observations=[],
        transport_diagnostics=[],
        latency_ms=(time.monotonic() - started_at) * 1000,
    )


def _trace_execution(
    tracer: Tracer,
    req: ExecuteRequest,
    result: ExecuteResult,
    *,
    parsed_result: Any,
    strategy: str,
) -> None:
    """Record invocation evidence, including predictions withheld by validation."""
    tracer.log(
        signature=req.signature,
        docstring=req.docstring,
        inputs=req.inputs,
        raw_text=result.raw_text,
        parsed_result=parsed_result,
        model=result.model,
        strategy=strategy,
        latency_ms=result.latency_ms,
        tokens_used=result.tokens_used,
        token_budget_used=result.token_budget_used,
        cached=result.cached,
        retries=result.retries,
        reasoning_requests=result.reasoning_requests,
        reasoning_observations=result.reasoning_observations,
        answer_format=result.answer_format,
        transport_diagnostics=result.transport_diagnostics,
        substrategy=result.substrategy,
        revised=result.revised,
        pre_revision_value=result.pre_revision_value,
        provider_finish_reason=result.provider_finish_reason,
        prompts=result.prompts,
        input_warnings=result.input_warnings,
        type_warnings=result.type_warnings,
        constraint_warnings=result.constraint_warnings,
    )
