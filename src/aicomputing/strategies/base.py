"""
Abstract strategy interface for AI Computing.

A strategy defines *how* to turn a specification (signature + docstring + inputs)
into a predicted output value. Different strategies may use different prompts,
reasoning approaches, or even generate and execute code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..backends.base import Backend, GenerateResponse
from ..request import ExecuteRequest

logger = logging.getLogger("aicomputing")

# Doubling retry config
_MAX_RETRIES = 5          # max doublings: 512→1024→2048→4096→8192→16384
_MAX_TOKEN_CAP = 32768    # never exceed this regardless of doublings


@dataclass
class StrategyResult:
    """Result from a strategy execution (before runtime validation)."""
    value: Any
    raw_text: str
    tokens_used: Optional[int] = None
    token_budget_used: Optional[int] = None
    retries: int = 0


class Strategy(ABC):
    """Abstract base for execution strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'direct', 'cot', 'code_gen'."""

    @abstractmethod
    def execute(
        self,
        request: ExecuteRequest,
        backend: Backend,
        feedback: Optional[dict] = None,
    ) -> StrategyResult:
        """
        Execute the AI function call using this strategy.

        Receives the full request and a backend to generate text.
        `feedback` is set on validation retries and contains the previous
        raw output and the validation error message:
        {"raw_output": str, "error": str}. Strategies append it to the
        prompt so the model can correct its answer.
        Returns a StrategyResult with the parsed value.
        """


def generate_with_retry(
    call: Callable[[int], GenerateResponse],
    parse: Callable[[str], Any],
    initial_budget: int,
    model_name: str = "",
) -> tuple[Any, GenerateResponse, int]:
    """
    Call the backend with doubling retries when the model hits the token limit.

    Args:
        call:           Function that takes max_new_tokens and returns GenerateResponse.
        parse:          Function that takes the response text and returns the parsed value.
                        Should raise ValueError or return None on failure.
        initial_budget: Starting token limit (512 for no-think, 8192 for think).
        model_name:     Used only for log messages.

    Returns:
        (parsed_value, final_response, retries) — retries is the number of
        times the budget was doubled (0 means it worked on the first attempt).
    """
    budget = initial_budget
    last_response = None

    for attempt in range(_MAX_RETRIES + 1):
        response = call(budget)
        last_response = response

        cutoff = response.finish_reason == "length"

        if not cutoff:
            try:
                value = parse(response.text)
                if attempt > 0:
                    logger.info(
                        "[retry] %s: succeeded at budget=%d (attempt %d)",
                        model_name, budget, attempt + 1,
                    )
                return value, response, attempt
            except (ValueError, TypeError):
                return None, response, attempt

        if budget >= _MAX_TOKEN_CAP:
            # Already tried at the cap — retrying at the same budget
            # would give the same truncated output.
            logger.error(
                "[retry] %s: still truncated at token cap %d, giving up.",
                model_name, _MAX_TOKEN_CAP,
            )
            return None, last_response, attempt

        new_budget = min(budget * 2, _MAX_TOKEN_CAP)
        logger.warning(
            "[retry] %s: hit token limit at budget=%d, retrying with budget=%d (attempt %d/%d)",
            model_name, budget, new_budget, attempt + 1, _MAX_RETRIES,
        )
        budget = new_budget

    logger.error(
        "[retry] %s: all %d retries exhausted. Last budget=%d, finish_reason=%s",
        model_name, _MAX_RETRIES, budget, getattr(last_response, "finish_reason", "?"),
    )
    return None, last_response, _MAX_RETRIES
