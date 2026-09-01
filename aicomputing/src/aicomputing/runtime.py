"""
VirtualRuntime — the central orchestrator for AI Computing.

Wires together a backend (how to talk to the LLM), a strategy (how to
turn a spec into an answer), and validation (type-checking + constraints).
Implements the operational semantics from the paper (rule E-AI):

    e ⇓ v    ⟨σ, d, v⟩ ⇓ r
    ─────────────────────────
         (σ, d)(e) ⇓ r
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional

from .backends import Backend, create_backend
from .cache import ResultCache
from .parsing import coerce_type
from .request import ExecuteRequest, ExecuteResult, FailureType
from .strategies import Strategy, create_strategy
from .strategies.base import CallStats, _accumulate
from .trace import Tracer
from .types import ParsedSignature, parse_signature
from .validation import (
    ValidationMode,
    AIConstraintError,
    AITypeError,
    validate_constraints,
    validate_inputs,
    validate_output,
)

logger = logging.getLogger("aicomputing")


class VirtualRuntime:
    """
    LLM-as-runtime facade.

    Example::

        vr = VirtualRuntime(backend="ollama", model="qwen3:4b", strategy="direct")
        result = vr.execute(ExecuteRequest(
            signature="add(x: int, y: int) -> int",
            docstring="Return the sum of x and y.",
            inputs={"x": 2, "y": 3},
            return_type=int,
        ))
        assert result.result == 5
    """

    def __init__(
        self,
        backend: str | Backend = "ollama",
        model: Optional[str] = None,
        strategy: str | Strategy = "direct",
        strict_types: bool = True,
        validation_mode: "ValidationMode | str | bool | None" = None,
        cache_size: int = 2048,
        trace_file: Optional[str] = None,
        **backend_kwargs,
    ) -> None:
        # Set up backend
        if isinstance(backend, str):
            bk_kwargs = dict(backend_kwargs)
            if model:
                bk_kwargs["model"] = model
            self._backend = create_backend(backend, **bk_kwargs)
        else:
            self._backend = backend

        # Set up strategy
        if isinstance(strategy, str):
            self._strategy = create_strategy(strategy)
        else:
            self._strategy = strategy

        self._strict_types = strict_types

        # Default structural-validation policy for calls that do not override
        # it. WARN preserves the permissive behaviour library and CLI callers
        # already had; the experiment constructs the runtime with STRICT.
        # strict_types is honoured when validation_mode is not given, so the
        # older keyword keeps working.
        self._validation_mode = ValidationMode.coerce(
            validation_mode,
            ValidationMode.WARN,
        )

        self._cache = ResultCache(maxsize=cache_size)
        self._tracer = Tracer(trace_file) if trace_file else None

        # Keep model name for reporting
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
        Execute an AI function call.

        Implements the E-AI evaluation rule:
        1. Parse and validate the signature + inputs  (T-AI type checking)
        2. Check the cache
        3. Run the strategy to get a predicted result
        4. Validate the output type and constraints
        5. Retry if validation fails (up to max_retries)
        6. Cache and return
        """
        t0 = time.monotonic()

        model = req.model or self._model
        strat_name = req.strategy or self._strategy.name
        strict = req.strict_types if req.strict_types is not None else self._strict_types

        # --- Step 1: Parse signature and validate inputs (T-AI) ---
        # In non-strict mode validate_inputs reports rather than raises, so
        # these warnings are the only signal that the caller supplied a
        # malformed task. Every benchmark runs non-strict, so they are
        # carried onto the result and into the trace rather than discarded.
        parsed_sig: Optional[ParsedSignature] = None
        input_warnings: List[str] = []
        try:
            parsed_sig = parse_signature(req.signature)
            input_warnings = validate_inputs(parsed_sig, req.inputs, strict=strict)
        except (ValueError, AITypeError) as e:
            logger.warning("Input validation: %s", e)
            if strict:
                raise
            input_warnings = [str(e)]

        # --- Step 2: Cache lookup ---
        cached = self._cache.get(
            req.signature, req.docstring, req.inputs, model, strat_name,
            think=req.think, temperature=req.temperature,
            top_p=req.top_p, max_new_tokens=req.max_new_tokens,
        )
        if cached is not None:
            # Return a copy so callers can't mutate the cached entry.
            hit = replace(
                cached,
                cached=True,
                input_warnings=input_warnings,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
            # Trace the hit as well. This path used to return before reaching
            # the tracer, so `cached` was only ever written as False and the
            # trace could not report a hit rate. tokens_used is reported as 0
            # because this invocation made no model call — the tokens the
            # answer originally cost are on the earlier uncached line, so a
            # naive sum over the trace stays correct.
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
                    # The ceiling the original call requested, carried over
                    # from the cached result: this invocation requested none.
                    token_budget_used=hit.token_budget_used,
                    cached=True,
                    retries=hit.retries,
                    think=req.think,
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

        # --- Step 3: Execute strategy (with retries) ---
        strategy = (
            create_strategy(strat_name) if strat_name != self._strategy.name
            else self._strategy
        )

        last_error: Optional[str] = None
        retries = 0
        feedback: Optional[Dict[str, Any]] = None

        # Resolved before the loop because coercion depends on it: STRICT
        # applies only meaning-preserving conversions, so the policy has to be
        # known while the value is being coerced, not merely afterwards.
        mode = ValidationMode.coerce(
            req.validation_mode,
            ValidationMode.coerce(req.strict_types, self._validation_mode),
        )
        strict = mode is ValidationMode.STRICT

        # Usage accumulated across EVERY attempt of this invocation.
        #
        # CallStats is created per strategy.execute(), so a validation retry
        # started a fresh one and the calls made by earlier attempts vanished
        # from the totals: a two-call strategy that was retried once reported
        # model_calls=2 when four calls had really been made and billed. The
        # reliability-vs-compute comparison depends on this number, and the
        # arms most likely to be retried are the ones whose cost is already
        # highest, so the error ran in the direction that flattered them.
        total_stats = CallStats()

        # The strategy's parser coerces before the runtime validates, so it
        # must see the same policy. Without this the parser resolved WARN from
        # a request that carried no explicit mode and repaired 3.7 into 3;
        # STRICT then validated the repaired value and passed it, defeating
        # the whole point of the strict condition.
        effective_req = replace(req, validation_mode=mode)

        for attempt in range(1 + max(0, req.max_retries)):
            strat_result = strategy.execute(effective_req, self._backend,
                                            feedback=feedback)
            _accumulate(total_stats, strat_result.call_stats)

            # --- Step 4: Coerce + Validate output ---
            type_warnings: List[str] = []
            constraint_warnings: List[str] = []

            # Determine the expected return type
            return_type = req.return_type
            if return_type is None and parsed_sig:
                return_type = parsed_sig.return_type

            # Coerce the result to match the expected type (e.g. list → tuple).
            # Under STRICT only conversions that preserve meaning are applied,
            # so the runtime cannot repair 3.7 into 3 and then report that the
            # model returned a structurally valid int.
            if return_type is not None:
                strat_result.value = coerce_type(
                    strat_result.value, return_type, strict=strict)

            # Type check (T-AI: result has type T_out)
            if return_type is not None:
                try:
                    type_warnings = validate_output(
                        strat_result.value, return_type, strict=False,
                    )
                except AITypeError as e:
                    type_warnings = [str(e)]

            # Constraint check
            if req.constraints:
                try:
                    constraint_warnings = validate_constraints(
                        strat_result.value, req.constraints, strict=False,
                    )
                except AIConstraintError as e:
                    constraint_warnings = [str(e)]

            # If validation passes (or we're out of retries), accept
            if not type_warnings and not constraint_warnings:
                break

            # Don't retry on token exhaustion — same budget will produce
            # the same empty result. Detect by: empty raw_text + high token count.
            if not strat_result.raw_text and strat_result.tokens_used:
                logger.info(
                    "Skipping retry: token budget exhausted (%d tokens used, "
                    "response was empty — model spent all tokens on thinking)",
                    strat_result.tokens_used,
                )
                break

            if attempt < req.max_retries:
                last_error = "; ".join(type_warnings + constraint_warnings)
                feedback = {
                    "raw_output": strat_result.raw_text,
                    "error": last_error,
                }
                retries += 1
                logger.info("Retry %d/%d: %s", retries, req.max_retries, last_error)

        latency = (time.monotonic() - t0) * 1000

        # --- Step 4b: apply the validation policy -------------------------
        #
        # The loop above deliberately collects warnings instead of raising, so
        # that a failed check can drive a retry. Once retries are exhausted the
        # policy decides what those warnings MEAN.
        #
        # Previously this step did not exist: validate_output was called with
        # strict=False unconditionally, so a value already known to violate the
        # declared return type was returned as a successful invocation and the
        # warning was the only trace. `strict_types=True` changed nothing about
        # output handling. Under STRICT the invalid value is now withheld --
        # the runtime does not hand back an object it has just determined does
        # not satisfy the contract.
        #
        # The failure is reported, not raised. Raising would abort the caller's
        # whole run over one bad example, which is the failure mode
        # bench_args.execute_resilient exists to prevent; a marked row lets the
        # cell finish and stay countable.
        validation_status = "valid"
        failure_type: Optional[str] = None
        final_value = strat_result.value

        if type_warnings:
            validation_status = "invalid_type"
        elif constraint_warnings:
            validation_status = "invalid_constraint"
        elif final_value is None and strat_result.exec_status is None:
            # Nothing was recoverable from the model's text. Distinct from a
            # wrong answer: the specification was never executed.
            validation_status = "unparsed"

        if validation_status != "valid":
            failure_type = {
                "invalid_type": FailureType.STRUCTURAL_VALIDATION,
                "invalid_constraint": FailureType.CONSTRAINT_VALIDATION,
                "unparsed": FailureType.PARSE,
            }[validation_status]
            if mode is ValidationMode.STRICT and validation_status != "unparsed":
                logger.info(
                    "STRICT validation: withholding %s output (%s)",
                    validation_status, "; ".join(type_warnings + constraint_warnings),
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
            # Cost accounting summed over every call the strategy made. Left
            # as None when the strategy did not report any (e.g. a result
            # built outside generate_with_retry), never defaulted to 0.
            # Accumulated across every attempt, not just the last one.
            input_tokens=total_stats.input_tokens,
            output_tokens=total_stats.output_tokens,
            total_tokens=total_stats.total_tokens,
            model_calls=total_stats.model_calls,
            parse_status=strat_result.parse_status,
            parse_error=strat_result.parse_error,
            requested_think=total_stats.requested_think,
            effective_think=total_stats.effective_think,
            thinking_honoured=total_stats.thinking_honoured,
            latency_ms=latency,
        )

        # --- Step 5: Cache (successful results only) ---
        # Caching parse failures or invalid outputs would poison every
        # future call with the same arguments.
        if not type_warnings and not constraint_warnings and strat_result.value is not None:
            self._cache.put(
                req.signature, req.docstring, req.inputs, model, strat_name, result,
                think=req.think, temperature=req.temperature,
                top_p=req.top_p, max_new_tokens=req.max_new_tokens,
            )

        # --- Step 6: Trace ---
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
                think=req.think,
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
