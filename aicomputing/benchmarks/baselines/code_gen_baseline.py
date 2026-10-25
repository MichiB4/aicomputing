"""Generate and execute Python implementations for benchmark comparison.

This evaluation-only strategy is supplied explicitly to the runtime and is
excluded from the specification-only oracle strategy registry. The model
receives the signature and docstring; the generated function receives inputs.
Its actual return value is used without a predicted-answer fallback.

Generated truncation, syntax errors, program errors and worker-wait timeouts
are scorable outcomes. Invalid declared headers and execution-machinery
failures propagate for the benchmark wrapper to classify as harness failures.
Provider exceptions also propagate for their existing classification.
Generation has no application output-token cap unless the request supplies
one; provider output and context limits still apply. Truncated source is
never executed, whether the request had an explicit token budget or not.

The restricted environment is not a security boundary. Results must support
multiprocessing serialization. See _sandbox_exec for the timeout's scope.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import logging
import math
import multiprocessing
import re
import textwrap
from typing import Any, Dict, Optional

from aicomputing.backends.base import Backend
from aicomputing.parsing import ParseOutcome
from aicomputing.prompting import build_codegen_prompt
from aicomputing.request import ExecuteRequest
from aicomputing.strategies.base import (
    CallStats, Strategy, StrategyResult, generate_with_retry,
)

logger = logging.getLogger("aicomputing")


class _GeneratedCodeError(RuntimeError):
    """The worker reported a failure in the generated program."""


class _GeneratedCodeTimeout(TimeoutError):
    """No generated outcome was ready within the execution wait."""


# Restricted builtins for sandboxed execution
_SAFE_BUILTINS = {
    "__build_class__",
    "abs", "all", "any", "bin", "bool", "bytes", "callable", "chr",
    "dict", "divmod", "enumerate", "filter", "float", "format",
    "frozenset", "getattr", "hash", "hex", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min", "next",
    "object", "oct", "ord", "pow", "print", "range", "repr", "reversed",
    "round", "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
    "True", "False", "None",
}


# Modules available through the restricted environment and import hook.
# Keep aligned with the names in prompting.build_codegen_prompt.
_ALLOWED_MODULES = (
    "math", "json", "re", "itertools", "functools",
    "collections", "string", "statistics",
)


def _build_safe_globals() -> Dict[str, Any]:
    """Build the restricted environment exposed through __builtins__.

    Include standard exception classes for ordinary raising and handling.
    Only the listed module roots are importable. This mapping is not a
    security boundary.
    """
    safe = {
        name: getattr(builtins, name)
        for name in _SAFE_BUILTINS
    }
    safe.update({
        name: value
        for name, value in vars(builtins).items()
        if isinstance(value, type) and issubclass(value, BaseException)
    })

    modules = {
        name: importlib.import_module(name)
        for name in _ALLOWED_MODULES
    }
    safe.update(modules)

    def _restricted_import(
        name, globals=None, locals=None, fromlist=(), level=0,
    ):
        if level != 0:
            raise ImportError(
                "Relative imports are unavailable in standalone generated code"
            )

        root = name.split(".")[0]
        if root not in modules:
            raise ImportError(
                f"import of {name!r} is not available in this sandbox; "
                f"allowed: {', '.join(_ALLOWED_MODULES)}"
            )

        return builtins.__import__(name, globals, locals, fromlist, level)

    safe["__import__"] = _restricted_import
    safe["__builtins__"] = safe
    return safe


def _extract_function_body(raw: str) -> str:
    """Remove output fences without stripping the first code line's indentation."""
    text = raw.strip("\r\n")
    text = re.sub(r"\A\s*```(?:python)?[ \t]*(?:\r?\n)?", "", text)
    text = re.sub(r"\s*```[ \t\r\n]*\Z", "", text)
    return text


def _parse_statements(source: str) -> tuple[list[ast.stmt], bool]:
    """Parse bare or indented statements; report whether a block was needed."""
    try:
        return ast.parse(source).body, False
    except IndentationError:
        block = ast.parse("if True:\n" + source).body[0]
        return block.body, True


def _assemble_function(body: str, signature: str, docstring: str,
                       func_name: str) -> str:
    """Keep a complete target definition or wrap the returned statements.

    Leading indentation is accepted as one enclosing block. Unindented
    complete definitions remain unchanged; indented definitions are rendered
    from their AST. Invalid source is passed to the compilation failure path.
    """
    try:
        statements, indented = _parse_statements(body)
    except SyntaxError:
        return _wrap_body(body, signature, docstring)
    defines = any(isinstance(node, ast.FunctionDef) and node.name == func_name
                  for node in statements)
    if defines:
        if indented:
            module = ast.Module(body=statements, type_ignores=[])
            return ast.unparse(module) + "\n"
        return body
    return _wrap_body(body, signature, docstring)


def _wrap_body(body: str, signature: str, docstring: str) -> str:
    """Wrap parsed statements while preserving nesting and string values."""
    header = f"def {signature}:\n    {docstring!r}\n"
    try:
        statements, _ = _parse_statements(body)
        function = ast.parse(header).body[0]
        function.body.extend(statements)
        return ast.unparse(function) + "\n"
    except SyntaxError:
        # Keep malformed output in the existing compile-and-score path.
        return header + textwrap.indent(body, "    ")


def _extract_func_name(signature: str) -> str:
    """Validate a Python function header and return its normalized name."""
    if not isinstance(signature, str):
        raise TypeError("signature must be a string")

    try:
        module = ast.parse(f"def {signature}:\n    pass\n")
        compile(module, filename="<codegen_signature>", mode="exec")
    except SyntaxError as exc:
        raise ValueError(
            f"Invalid Python function signature: {signature!r}"
        ) from exc

    if (
        len(module.body) != 1
        or not isinstance(module.body[0], ast.FunctionDef)
        or len(module.body[0].body) != 1
        or not isinstance(module.body[0].body[0], ast.Pass)
    ):
        raise ValueError(f"Invalid Python function signature: {signature!r}")

    return module.body[0].name


def _describe_exception(exc: BaseException) -> str:
    """Keep error reporting usable even when an exception cannot be formatted."""
    try:
        return f"{type(exc).__name__}: {exc}"
    except BaseException:
        return f"{type(exc).__name__}: error message unavailable"


def _exec_worker(
    func_code: str,
    func_name: str,
    inputs: Dict[str, Any],
    conn,
) -> None:
    """Execute generated code in a child process and send its outcome."""
    try:
        try:
            namespace: Dict[str, Any] = {
                "__builtins__": _build_safe_globals(),
                "__name__": "__aicomputing_generated__",
            }
        except Exception as exc:
            conn.send(("harness_error", _describe_exception(exc)))
            return

        try:
            compiled = compile(
                func_code, filename="<codegen_baseline>", mode="exec",
            )
            # Functions must see definitions made by the same generated code.
            exec(compiled, namespace)
            if func_name not in namespace:
                message = (
                    "error",
                    f"Generated code did not define function {func_name!r}",
                )
            else:
                message = ("ok", namespace[func_name](**inputs))
        except BaseException as exc:
            message = ("error", _describe_exception(exc))

        # Transport errors must not be reported as generated-program errors.
        conn.send(message)
    finally:
        conn.close()


class CodeGenBaseline(Strategy):
    """
    Comparison baseline: generate code, execute it, return the real output.

    This is NOT AI computing. It is the traditional LLM-for-code approach.
    Included for benchmark comparison only.
    """

    def __init__(self, timeout: float = 10.0) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a positive, finite number of seconds")

        try:
            timeout = float(timeout)
        except OverflowError as exc:
            raise ValueError(
                "timeout must be a positive, finite number of seconds"
            ) from exc

        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive, finite number of seconds")

        self._timeout = timeout

    @property
    def name(self) -> str:
        return "code_gen_baseline"

    def execute(
        self,
        request: ExecuteRequest,
        backend: Backend,
        feedback: Optional[dict] = None,
    ) -> StrategyResult:
        func_name = _extract_func_name(request.signature)
        prompt = build_codegen_prompt(
            signature=request.signature,
            docstring=request.docstring,
        )
        if feedback:
            prompt = (
                f"{prompt}\n\n"
                f"Previous generated response:\n{feedback.get('raw_output', '')}\n\n"
                f"Validation error: {feedback.get('error', '')}\n"
                "Correct the implementation to satisfy the original specification "
                "and address the validation error. Return Python code only, "
                "using the implementation format requested above."
            )
        initial_budget = request.max_new_tokens

        def call(budget: Optional[int]):
            return backend.generate(
                prompt, model=request.model, temperature=request.temperature,
                top_p=request.top_p, max_new_tokens=budget,
            )

        stats = CallStats()
        body, response, retries, budget = generate_with_retry(
            call=call, parse=_extract_function_body, initial_budget=initial_budget,
            stats=stats, model_name=request.model or backend.default_model,
        )
        truncated = response.finish_reason == "length"
        value = None
        exec_status = "truncated" if truncated else "ok"
        error_message = (
            "Generation ended at an output or context limit."
            if truncated else None
        )

        # Never execute a prefix still marked as truncated.
        if not truncated:
            func_code = _assemble_function(
                body, request.signature, request.docstring, func_name,
            )
            try:
                compile(func_code, filename="<codegen_baseline>", mode="exec")
            except SyntaxError as exc:
                exec_status = "syntax_error"
                error_message = f"Generated code has a syntax error at line {exc.lineno}: {exc.msg}"
                logger.info("[codegen] generated code does not parse; scored as wrong")
            if exec_status == "ok":
                try:
                    value = self._sandbox_exec(func_code, func_name, request.inputs)
                except _GeneratedCodeTimeout:
                    exec_status = "timeout"
                    error_message = f"Generated code exceeded the {self._timeout}s execution timeout."
                    logger.info("[codegen] generated code timed out; scored as wrong")
                except _GeneratedCodeError as exc:
                    exec_status = "runtime_error"
                    error_message = f"Generated code failed: {exc}"
                    logger.info("[codegen] %s; scored as wrong", error_message)

        # Usage covers every returned response; raw text is the final response.
        return StrategyResult(
            value=value,
            prompts=[prompt],
            raw_text=f"[codegen {exec_status}] {response.text}",
            tokens_used=response.tokens_used,
            provider_finish_reason=response.provider_finish_reason,
            token_budget_used=budget,
            retries=retries,
            exec_status=exec_status,
            call_stats=stats,
            parse_status=ParseOutcome.FAILED if exec_status != "ok" else None,
            parse_error=error_message,
        )

    def _sandbox_exec(
        self,
        func_code: str,
        func_name: str,
        inputs: Dict[str, Any],
    ) -> Any:
        """Run generated source in a child process and receive its outcome.

        The timeout bounds waiting for result-pipe readiness after start()
        returns. If this wait expires, an active worker is terminated and a
        late generated outcome counts as _GeneratedCodeTimeout. Process
        startup, receiving the complete payload and cleanup are not bounded
        by this wait.

        Explicit worker program errors received within the wait become
        _GeneratedCodeError. Setup, startup and transport errors propagate
        as harness failures, including after the wait expires.
        """
        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
        started = False
        try:
            proc = multiprocessing.Process(
                target=_exec_worker,
                args=(func_code, func_name, inputs, child_conn),
            )
            proc.start()
            started = True
            child_conn.close()  # only the child keeps the sending end open
            wait_expired = not parent_conn.poll(self._timeout)
            if wait_expired:
                if proc.is_alive():
                    raise _GeneratedCodeTimeout(
                        f"Generated code exceeded the {self._timeout}s timeout"
                    )
                if not parent_conn.poll(0.0):
                    raise RuntimeError("Execution worker exited without returning a result")
            status, payload = parent_conn.recv()
        except EOFError as exc:
            raise RuntimeError("Execution worker exited without returning a result") from exc
        finally:
            child_conn.close()
            parent_conn.close()
            if started:
                if proc.is_alive():
                    proc.terminate()
                proc.join()

        if wait_expired and status in ("ok", "error"):
            raise _GeneratedCodeTimeout(
                f"Generated code exceeded the {self._timeout}s timeout"
            )
        if status == "error":
            raise _GeneratedCodeError(payload)
        if status == "harness_error":
            raise RuntimeError(f"Execution worker setup failed: {payload}")
        if status == "ok":
            return payload
        raise RuntimeError(f"Unknown execution worker status: {status!r}")
