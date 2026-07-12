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
from .request import ExecuteRequest, ExecuteResult
from .strategies import Strategy, create_strategy
from .trace import Tracer
from .types import ParsedSignature, parse_signature
from .validation import (
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
        parsed_sig: Optional[ParsedSignature] = None
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
        )
        if cached is not None:
            # Return a copy so callers can't mutate the cached entry.
            hit = replace(
                cached,
                cached=True,
                latency_ms=(time.monotonic() - t0) * 1000,
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

        for attempt in range(1 + max(0, req.max_retries)):
            strat_result = strategy.execute(req, self._backend, feedback=feedback)

            # --- Step 4: Coerce + Validate output ---
            type_warnings: List[str] = []
            constraint_warnings: List[str] = []

            # Determine the expected return type
            return_type = req.return_type
            if return_type is None and parsed_sig:
                return_type = parsed_sig.return_type

            # Coerce the result to match the expected type (e.g. list → tuple)
            # This handles cases where JSON can't represent certain Python types.
            if return_type is not None:
                strat_result.value = coerce_type(strat_result.value, return_type)

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

        result = ExecuteResult(
            result=strat_result.value,
            raw_text=strat_result.raw_text,
            model=model,
            strategy=strategy.name,
            type_warnings=type_warnings,
            constraint_warnings=constraint_warnings,
            retries=retries,
            token_retries=strat_result.retries,
            tokens_used=strat_result.tokens_used,
            latency_ms=latency,
        )

        # --- Step 5: Cache (successful results only) ---
        # Caching parse failures or invalid outputs would poison every
        # future call with the same arguments.
        if not type_warnings and not constraint_warnings and strat_result.value is not None:
            self._cache.put(
                req.signature, req.docstring, req.inputs, model, strat_name, result,
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
                cached=False,
                retries=retries,
                think=req.think,
                type_warnings=type_warnings,
                constraint_warnings=constraint_warnings,
            )

        return result
