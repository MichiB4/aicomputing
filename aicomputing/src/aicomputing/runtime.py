"""Coordinate inference, validation, retries, caching, and tracing."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backends import Backend, create_backend
from .backends.base import _close_after_error
from .answer_protocol import ANSWER_FORMAT
from .cache import ResultCache
from .parsing import coerce_type
from .request import ExecuteRequest, ExecuteResult, FailureType
from .status import ParseStatus, ValidationStatus
from .strategies import Strategy, create_strategy
from .strategies.base import CallStats, StrategyResult, _accumulate
from .trace import Tracer
from .types import parse_signature
from .validation import (
    ValidationMode,
    validate_constraints,
    validate_inputs,
    validate_nonblank_text,
    validate_output,
)

logger = logging.getLogger("aicomputing")


class VirtualRuntime:
    """
    Produce function outputs from specifications and concrete inputs.

    Prediction strategies return model-produced values. The optional
    code_execute strategy generates and runs an implementation locally.

    A backend must be supplied explicitly, by name or as a Backend instance.
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
            ))
        print(result.result)  # Expected answer: 5; predictions may be incorrect.
    """

    def __init__(
        self,
        backend: str | Backend,
        model: Optional[str] = None,
        strategy: str | Strategy = "direct",
        validation_mode: ValidationMode | str | None = None,
        cache_size: int = 0,
        trace_file: str | Path | None = None,
        **backend_kwargs,
    ) -> None:
        # Validate caller options before constructing or borrowing a backend.
        if not isinstance(backend, (str, Backend)):
            raise TypeError("backend must be a backend name or a Backend instance")
        validate_nonblank_text(model, "model", optional=True)
        if trace_file is not None and not isinstance(trace_file, (str, Path)):
            raise TypeError("trace_file must be a string, Path, or None")
        if isinstance(trace_file, str):
            validate_nonblank_text(trace_file, "trace_file")
        if not isinstance(strategy, (str, Strategy)):
            raise TypeError("strategy must be a strategy name or a Strategy instance")
        if isinstance(strategy, str):
            validate_nonblank_text(strategy, "strategy")
        self._validation_mode = ValidationMode.coerce(validation_mode)
        if not isinstance(backend, str) and backend_kwargs:
            raise TypeError("Backend options require a named backend, not an existing instance")
        # Named backends belong to this runtime; supplied instances are borrowed.
        self._owns_backend = isinstance(backend, str)
        self._closed = False
        if self._owns_backend:
            if model is not None:
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
            validate_nonblank_text(self._strategy.name, "strategy")

            self._cache = ResultCache(maxsize=cache_size)
            self._tracer = Tracer(trace_file) if trace_file is not None else None

            self._model = self._backend.default_model if model is None else model
            validate_nonblank_text(self._model, "model")
        except BaseException:
            if self._owns_backend:
                _close_after_error(self._backend.close)
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
        """Close resources, preserving an active error if cleanup also fails."""
        if exc_type is None:
            self.close()
        else:
            _close_after_error(self.close)

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

    def _prepare_cache(
        self, req: ExecuteRequest, *, model: str, strategy_name: str,
        strategy: Strategy, mode: ValidationMode,
    ) -> Optional[Dict[str, Any]]:
        """Return shared lookup/insertion arguments, or None to bypass caching."""
        if self._cache.maxsize == 0:
            return None
        if strategy_name == self._strategy.name and not self._cache_default_strategy:
            return None
        backend_state = self._backend.cache_state()
        # Active predicates lack safe cache identities; skipped checks do not affect caching.
        if backend_state is None or (mode is not ValidationMode.NONE and req.constraints):
            return None
        return {
            "signature": req.signature,
            "docstring": req.docstring,
            "inputs": req.inputs,
            "model": model,
            "strategy": strategy_name,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_new_tokens": req.max_new_tokens,
            "context": {
                "backend_class": (
                    type(self._backend).__module__,
                    type(self._backend).__qualname__,
                ),
                "backend_state": backend_state,
                "validation_mode": mode.value,
                "max_retries": req.max_retries,
                "strategy_class": (
                    type(strategy).__module__,
                    type(strategy).__qualname__,
                ),
                "strategy_state": strategy.cache_state(),
            },
        }

    def execute(self, req: ExecuteRequest) -> ExecuteResult:
        """Validate inputs, predict or reuse an output, and apply output checks.

        Failed checks may retry; eligible outputs are cached and traced.
        Strict input errors and backend exceptions may propagate.
        Predictions are not guaranteed correct.
        """
        if self._closed:
            raise RuntimeError("VirtualRuntime is closed")
        t0 = time.monotonic()
        req.validate_settings()

        model = self._model if req.model is None else req.model
        strat_name = self._strategy.name if req.strategy is None else req.strategy
        validate_nonblank_text(model, "model")
        validate_nonblank_text(strat_name, "strategy")
        parsed_sig = parse_signature(req.signature)
        validate_inputs(parsed_sig, req.inputs)

        # Resolve output requirements before cache lookup to prevent bypasses.
        strategy = (
            create_strategy(strat_name)
            if strat_name != self._strategy.name
            else self._strategy
        )
        mode = ValidationMode.coerce(req.validation_mode, self._validation_mode)
        return_type = parsed_sig.return_type

        cache_settings = self._prepare_cache(
            req, model=model, strategy_name=strat_name, strategy=strategy, mode=mode,
        )
        cached = self._cache.get(**cache_settings) if cache_settings is not None else None
        if cached is not None:
            hit = _cached_invocation(cached, t0)

            # Record cache hits too; they consume no new inference tokens.
            if self._tracer:
                self._tracer.log(req, hit)
            return hit

        retries = 0
        feedback: Optional[Dict[str, Any]] = None

        total_stats = CallStats()

        # Pass resolved settings to generation, parsing, and answer selection.
        effective_req = replace(
            req, model=model, validation_mode=mode,
        )

        validation_retries = req.max_retries if strategy.supports_validation_retries else 0
        for attempt in range(1 + validation_retries):
            strat_result = strategy.execute(
                effective_req,
                self._backend,
                feedback=feedback,
            )
            _accumulate(total_stats, strat_result.call_stats)

            checked = _check_prediction(req, strat_result, return_type, mode)

            # Retry failed recovery as well as type and constraint failures.
            if not (
                checked.recovery_failed
                or checked.type_warnings
                or checked.constraint_warnings
            ):
                break

            if attempt < validation_retries:
                errors = checked.type_warnings + checked.constraint_warnings
                if checked.recovery_failed:
                    errors.insert(
                        0,
                        strat_result.execution_error or strat_result.parse_error
                        or "Could not recover an output value from the model response.",
                    )
                last_error = "; ".join(errors)
                feedback = {
                    "raw_output": (
                        strat_result.raw_text if strat_result.feedback_text is None
                        else strat_result.feedback_text
                    ),
                    "error": last_error,
                }
                retries += 1
                logger.info(
                    "Validation retry %d/%d: %s",
                    retries,
                    req.max_retries,
                    last_error,
                )

        result = _build_result(
            strat_result, checked, total_stats, model=model, strategy_name=strategy.name,
            mode=mode, retries=retries, latency_ms=(time.monotonic() - t0) * 1000,
        )

        # Cache only valid predictions from eligible requests. ExecuteResult
        # distinguishes a cached None prediction from a missing cache entry.
        if cache_settings is not None and result.validation_status in (ValidationStatus.VALID, ValidationStatus.NOT_CHECKED):
            self._cache.put(result=result, **cache_settings)

        # Keep the recovered prediction in the trace even if STRICT withheld it.
        if self._tracer:
            self._tracer.log(req, result, parsed_result=strat_result.value)

        return result

@dataclass
class _PredictionCheck:
    """Runtime checks for one strategy attempt, before retry selection."""

    recovery_failed: bool
    type_warnings: List[str]
    constraint_warnings: List[str]


def _check_prediction(
    req: ExecuteRequest,
    strat_result: StrategyResult,
    return_type: Any,
    mode: ValidationMode,
) -> _PredictionCheck:
    """Coerce and check one recovered prediction, preserving failure precedence."""
    # A successful worker run supplies a value directly, without prediction parsing.
    # Prediction parsing distinguishes failed recovery from a valid None answer.
    if strat_result.execution_details is not None:
        recovery_failed = strat_result.execution_details["status"] != "ok"
    else:
        # An explicit parsing failure takes precedence over any attached value.
        recovery_failed = (
            strat_result.parse_status == ParseStatus.FAILED
            or (
                strat_result.value is None
                and strat_result.parse_status not in (ParseStatus.PARSED, ParseStatus.FALLBACK)
            )
        )

    type_warnings: List[str] = []
    constraint_warnings: List[str] = []

    if mode is ValidationMode.NONE:
        return _PredictionCheck(recovery_failed, type_warnings, constraint_warnings)

    # Both policies use the same conversions and collect warnings before retries.
    if return_type is not None:
        strat_result.value = coerce_type(strat_result.value, return_type)

        type_warnings = validate_output(
            strat_result.value,
            return_type,
            strict=False,
        )

    if req.constraints and not recovery_failed and not type_warnings:
        constraint_warnings = validate_constraints(
            strat_result.value,
            req.constraints,
            strict=False,
        )
    return _PredictionCheck(
        recovery_failed=recovery_failed,
        type_warnings=type_warnings,
        constraint_warnings=constraint_warnings,
    )


def _build_result(
    strat_result: StrategyResult,
    checked: _PredictionCheck,
    total_stats: CallStats,
    *, model: str, strategy_name: str, mode: ValidationMode,
    retries: int, latency_ms: float,
) -> ExecuteResult:
    """Assign the final outcome and combine attempt evidence with invocation usage."""
    # After retries, STRICT withholds invalid outputs; WARN retains them.
    validation_status = ValidationStatus.NOT_CHECKED if mode is ValidationMode.NONE else ValidationStatus.VALID
    failure_type: Optional[str] = None
    final_value = None if checked.recovery_failed else strat_result.value

    # Recovery failure takes precedence over incidental validation errors.
    # A successfully parsed None is instead checked against its output type.
    if checked.recovery_failed:
        validation_status = (
            ValidationStatus.EXECUTION_FAILED if strat_result.execution_details is not None else ValidationStatus.UNPARSED
        )
    elif checked.type_warnings:
        validation_status = ValidationStatus.INVALID_TYPE
    elif checked.constraint_warnings:
        validation_status = ValidationStatus.INVALID_CONSTRAINT

    if validation_status not in (ValidationStatus.VALID, ValidationStatus.NOT_CHECKED):
        failure_type = {
            ValidationStatus.INVALID_TYPE: FailureType.STRUCTURAL_VALIDATION,
            ValidationStatus.INVALID_CONSTRAINT: FailureType.CONSTRAINT_VALIDATION,
            ValidationStatus.UNPARSED: FailureType.PARSE,
            ValidationStatus.EXECUTION_FAILED: FailureType.EXECUTION,
        }[validation_status]
        if mode is ValidationMode.STRICT and validation_status is not ValidationStatus.UNPARSED:
            logger.info(
                "STRICT validation: withholding %s output (%s)",
                validation_status,
                "; ".join(checked.type_warnings + checked.constraint_warnings),
            )
            final_value = None

    return ExecuteResult(
        result=final_value,
        validation_status=validation_status,
        failure_type=failure_type,
        raw_text=strat_result.raw_text,
        model=model,
        strategy=strategy_name,
        substrategy=strat_result.substrategy,
        revised=strat_result.revised,
        pre_revision_value=strat_result.pre_revision_value,
        provider_finish_reason=strat_result.provider_finish_reason,
        prompts=strat_result.prompts,
        type_warnings=checked.type_warnings,
        constraint_warnings=checked.constraint_warnings,
        retries=retries,
        # Expose a component total only when every recorded call supplied it.
        input_tokens=total_stats.input_tokens if total_stats.input_complete else None,
        output_tokens=total_stats.output_tokens if total_stats.output_complete else None,
        total_tokens=total_stats.total_tokens,
        model_calls=total_stats.model_calls,
        parse_status=strat_result.parse_status,
        parse_error=strat_result.parse_error,
        verdict_parse_status=strat_result.verdict_parse_status,
        verdict_parse_error=strat_result.verdict_parse_error,
        verdict_finish_reason=strat_result.verdict_finish_reason,
        execution_error=strat_result.execution_error,
        reasoning_requests=total_stats.reasoning_requests,
        reasoning_observations=total_stats.reasoning_observations,
        answer_format=strat_result.answer_format or ANSWER_FORMAT,
        transport_diagnostics=total_stats.transport_diagnostics,
        latency_ms=latency_ms,
        generated_code=strat_result.generated_code,
        execution_details=strat_result.execution_details,
    )


def _cached_invocation(
    cached: ExecuteResult, started_at: float,
) -> ExecuteResult:
    """Reuse a prediction while recording zero new inference work."""
    return replace(
        cached,
        cached=True,
        retries=0,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        model_calls=0,
        reasoning_requests=[],
        reasoning_observations=[],
        transport_diagnostics=[],
        latency_ms=(time.monotonic() - started_at) * 1000,
    )
