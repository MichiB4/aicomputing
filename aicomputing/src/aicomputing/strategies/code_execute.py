"""Generate one Python implementation and run it in a standard Python worker."""
from __future__ import annotations

import math
from typing import Callable, Optional

from ..backends.base import Backend
from ..code_executor import execute_code_with_retries, extract_code
from ..request import ExecuteRequest
from .base import CallStats, Strategy, StrategyResult
from .code_prompt import ANSWER_FORMAT, build_prompt


class CodeExecutionError(RuntimeError):
    """The execution environment failed, independently of generated-code quality."""

    def __init__(self, generated_code, execution_details):
        diagnostic = ": ".join(detail for detail in (
            execution_details.get("error_type"), execution_details.get("error_message")
        ) if detail)
        super().__init__("Python worker failed" + (": " + diagnostic if diagnostic else ""))
        self.generated_code = generated_code
        self.execution_details = execution_details


class CodeExecuteStrategy(Strategy):
    """Generate once; retry only execution timeouts using identical source/input.

    ``on_generated_code`` can persist extracted source before execution begins.
    No expected answer or benchmark data is accepted by this strategy.
    Current configuration is validated at construction and before generation.
    """

    supports_validation_retries = False

    def __init__(
        self, *, execution_timeout: float = 120.0, execution_retries: int = 2,
        task_mode: str = "function",
        on_generated_code: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.execution_timeout = execution_timeout
        self.execution_retries = execution_retries
        self.task_mode = task_mode
        self.on_generated_code = on_generated_code
        self._validate_settings()

    def _validate_settings(self) -> None:
        """Validate current settings without generation or checkpoint callbacks."""
        if (
            isinstance(self.execution_timeout, bool)
            or not isinstance(self.execution_timeout, (int, float))
            or not math.isfinite(self.execution_timeout)
            or self.execution_timeout <= 0
        ):
            raise ValueError("execution_timeout must be finite and positive")
        if type(self.execution_retries) is not int or self.execution_retries < 0:
            raise ValueError("execution_retries must be a nonnegative integer")
        if self.task_mode not in ("function", "word_problem"):
            raise ValueError("task_mode must be 'function' or 'word_problem'")
        if self.on_generated_code is not None and not callable(self.on_generated_code):
            raise TypeError("on_generated_code must be callable or None")

    @property
    def name(self) -> str:
        return "code_execute"

    def execute(self, request: ExecuteRequest, backend: Backend,
                feedback: Optional[dict] = None) -> StrategyResult:
        self._validate_settings()
        prompt = build_prompt(request.signature, request.docstring, request.inputs,
                              task_mode=self.task_mode)
        response = backend.generate_chat(
            [{"role": "user", "content": prompt}], model=request.model,
            temperature=request.temperature, top_p=request.top_p,
            max_new_tokens=request.max_new_tokens,
        )
        stats = CallStats().record(response)
        code = None
        execution = dict(status="format_error", value=None, error_type="ValueError", elapsed_seconds=0.0,
                         error_message="Model response was truncated at the output-token limit.", error_line=None)
        # An explicit cap is never expanded, and truncated source is not run.
        if response.finish_reason != "length":
            try:
                code = extract_code(response.parsing_text)
            except ValueError as exc:
                execution["error_message"] = str(exc)
            else:
                if self.on_generated_code is not None:
                    self.on_generated_code(code)
                execution = execute_code_with_retries(
                    code, request.signature, request.inputs,
                    timeout_seconds=self.execution_timeout,
                    timeout_retries=self.execution_retries,
                )
                if execution["status"] == "harness_error":
                    raise CodeExecutionError(code, execution)
        ok = execution["status"] == "ok"
        execution_error = None
        if not ok:
            execution_error = execution["status"] + ": " + (execution["error_type"] or "unknown")
            if execution["error_message"]:
                execution_error += ": " + execution["error_message"]
            if execution["error_line"] is not None:
                execution_error += f" (generated code, line {execution['error_line']})"
        return StrategyResult(
            value=execution["value"], raw_text=response.text, prompts=[prompt],
            provider_finish_reason=response.provider_finish_reason, call_stats=stats,
            execution_error=execution_error,
            generated_code=code, execution_details=execution, answer_format=ANSWER_FORMAT,
        )
