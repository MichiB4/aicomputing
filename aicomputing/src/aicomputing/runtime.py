"""
VirtualRuntime — the central orchestrator for AI Computing.

Coordinates a backend for model inference, a strategy for predicting outputs
from specifications and inputs, and type and constraint validation.
Also manages execution retries, result caching, and tracing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional

from .backends import Backend, create_backend
from .cache import ResultCache
from .parsing import ParseOutcome, coerce_type
from .request import ExecuteRequest, ExecuteResult, FailureType
from .strategies import Strategy, create_strategy
from .strategies.base import CallStats, _accumulate
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

    Example::

        vr = VirtualRuntime(backend="ollama", model="qwen3:4b", strategy="direct")
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
        # Set up backend
        if isinstance(backend, str):
            if model:
                backend_kwargs["model"] = model
            self._backend = create_backend(backend, **backend_kwargs)
        else:
            self._backend = backend

        # Named strategies have known configuration; custom instances may
        # depend on external state that cannot safely be represented in a key.
        self._cache_default_strategy = isinstance(strategy, str)
        # Set up strategy
        if isinstance(strategy, str):
            self._strategy = create_strategy(strategy)
        else:
            self._strategy = strategy

        self._strict_types = strict_types

        # Default output-validation policy; requests can override it.
        # None selects WARN independently of the input strict_types setting.
        self._validation_mode = ValidationMode.coerce(
            validation_mode,
            ValidationMode.WARN,
        )

        self._cache = ResultCache(maxsize=cache_size)
        self._tracer = Tracer(trace_file) if trace_file else None

        # Default model for requests that do not specify one.
        self._model = model or self._backend.default_model

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
        """
        Predict a function output from its specification and concrete inputs.

        Checks inputs and returns a cached result when available. Otherwise,
        runs the strategy, coerces and validates its output, and may retry
        after validation failures. Caches eligible results and optionally
        records a trace before returning.

        Strict input-validation errors and backend exceptions may propagate.
        A returned prediction is not guaranteed to be correct.
        """
        t0 = time.monotonic()
        req.validate_limits()

        # Capability restrictions apply under both input-validation policies.
        validate_media_inputs(req.inputs)

        model = req.model or self._model
        strat_name = req.strategy or self._strategy.name
        input_strict = (
            req.strict_types
            if req.strict_types is not None
            else self._strict_types
        )

        # Parse the signature and validate inputs.
        # Permissive input checking records warnings on the result and trace.
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

        # Resolve execution requirements before cache lookup. A cache hit must
        # not bypass output-policy validation or changed output requirements.
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
        return_type = req.return_type
        if return_type is None and parsed_sig is not None:
            return_type = parsed_sig.return_type

        validate_type_annotation(return_type)

        # Arbitrary predicates cannot be identified safely from their names
        # or mutable closure state, so constrained requests always run fresh.
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
            # The cache returns an isolated snapshot. Call records, usage and
            # retry counts describe this hit.
            hit = replace(
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
                latency_ms=(time.monotonic() - t0) * 1000,
            )

            # Record cache hits too; they consume no new inference tokens.
            if self._tracer:
                self._tracer.log(
                    signature=req.signature,
                    docstring=req.docstring,
                    inputs=req.inputs,
                    raw_text=hit.raw_text,
                    parsed_result=hit.result,
                    model=model,
                    strategy=strat_name,
                    latency_ms=hit.latency_ms,
                    tokens_used=0,
                    # This invocation requested no generation budget.
                    token_budget_used=hit.token_budget_used,
                    cached=True,
                    retries=hit.retries,
                    reasoning_requests=hit.reasoning_requests,
                    reasoning_observations=hit.reasoning_observations,
                    substrategy=hit.substrategy,
                    revised=hit.revised,
                    pre_revision_value=hit.pre_revision_value,
                    provider_finish_reason=hit.provider_finish_reason,
                    exec_status=hit.exec_status,
                    prompts=hit.prompts,
                    input_warnings=input_warnings,
                    type_warnings=hit.type_warnings,
                    constraint_warnings=hit.constraint_warnings,
                )
            return hit

        # Initialise validation-retry state.
        retries = 0
        feedback: Optional[Dict[str, Any]] = None

        # Accumulate reported usage across all strategy attempts.
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

            execution_failure = {
                "truncated": FailureType.GENERATED_CODE_TRUNCATED,
                "syntax_error": FailureType.GENERATED_CODE_SYNTAX,
                "runtime_error": FailureType.GENERATED_CODE_RUNTIME,
                "timeout": FailureType.GENERATED_CODE_TIMEOUT,
            }.get(strat_result.exec_status)

            # Capture missing results before permissive coercion. A failed
            # execution is not a successful return of None.
            parse_failed = (
                execution_failure is not None
                or (
                    strat_result.exec_status is None
                    and strat_result.value is None
                    and strat_result.parse_status not in (
                        ParseOutcome.PARSED,
                        ParseOutcome.FALLBACK,
                    )
                )
            )

            # Each attempt starts with fresh output warnings.
            type_warnings: List[str] = []
            constraint_warnings: List[str] = []

            # Apply the same limited conversions under WARN and STRICT.
            if return_type is not None:
                strat_result.value = coerce_type(
                    strat_result.value,
                    return_type,
                    strict=output_strict,
                )

            # Collect failures before deciding whether to retry or withhold.
            if return_type is not None:
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

            # Retry failed recovery as well as type and constraint failures.
            if not parse_failed and not type_warnings and not constraint_warnings:
                break

            if attempt < req.max_retries:
                errors = type_warnings + constraint_warnings
                if parse_failed:
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

        # Apply output policy after the retry loop ends.
        # STRICT withholds outputs that fail type or constraint validation.
        # WARN returns those outputs with diagnostics.
        validation_status = "valid"
        failure_type: Optional[str] = None
        final_value = None if parse_failed else strat_result.value

        # Recovery failure takes precedence over incidental validation errors.
        # A successfully parsed None is instead checked against its output type.
        if parse_failed:
            validation_status = "unparsed"
        elif type_warnings:
            validation_status = "invalid_type"
        elif constraint_warnings:
            validation_status = "invalid_constraint"

        if validation_status != "valid":
            failure_type = {
                "invalid_type": FailureType.STRUCTURAL_VALIDATION,
                "invalid_constraint": FailureType.CONSTRAINT_VALIDATION,
                "unparsed": FailureType.PARSE,
            }[validation_status]
            if execution_failure is not None:
                failure_type = execution_failure

            if mode is ValidationMode.STRICT and validation_status != "unparsed":
                logger.info(
                    "STRICT validation: withholding %s output (%s)",
                    validation_status,
                    "; ".join(type_warnings + constraint_warnings),
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
            exec_status=strat_result.exec_status,
            prompts=strat_result.prompts,
            input_warnings=input_warnings,
            type_warnings=type_warnings,
            constraint_warnings=constraint_warnings,
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
            latency_ms=latency,
        )

        # Cache only valid predictions from eligible requests. ExecuteResult
        # distinguishes a cached None prediction from a missing cache entry.
        if (
            cache_allowed
            and validation_status == "valid"
            and result.exec_status in (None, "ok")
        ):
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

        # Record the uncached invocation.
        if self._tracer:
            self._tracer.log(
                signature=req.signature,
                docstring=req.docstring,
                inputs=req.inputs,
                raw_text=strat_result.raw_text,
                parsed_result=strat_result.value,
                model=model,
                strategy=strategy.name,
                latency_ms=latency,
                tokens_used=strat_result.tokens_used,
                token_budget_used=strat_result.token_budget_used,
                cached=False,
                retries=retries,
                reasoning_requests=result.reasoning_requests,
                reasoning_observations=result.reasoning_observations,
                substrategy=strat_result.substrategy,
                revised=strat_result.revised,
                pre_revision_value=strat_result.pre_revision_value,
                provider_finish_reason=strat_result.provider_finish_reason,
                exec_status=strat_result.exec_status,
                prompts=strat_result.prompts,
                input_warnings=input_warnings,
                type_warnings=type_warnings,
                constraint_warnings=constraint_warnings,
            )

        return result
