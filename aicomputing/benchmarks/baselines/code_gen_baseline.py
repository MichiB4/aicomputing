"""
Code generation + execution baseline.

THIS IS NOT AN AI COMPUTING STRATEGY. It represents the *traditional*
approach: generate code from a spec, execute it, return the real output.
It lives in benchmarks/baselines/ because it exists solely to compare
against true AI computing (spec-only prediction).

The comparison answers the question: "When does predicting the output
from a spec alone match or beat generating-and-running actual code?"

Scoring rule (thesis §5.5)
--------------------------
A generation that does not execute scores as **wrong**. There is no fallback
to reading an answer out of the generated source text: doing so would let the
baseline borrow the spec-only mechanism it is meant to be compared against,
understating the very gap §6.3 reports. The failure category is recorded in
`StrategyResult.exec_status` so the execution-failure rate is reported
alongside accuracy rather than folded into it:

    ok            the generated function ran and returned a value
    syntax_error  the generated code does not parse
    runtime_error it raised, or defined no function of the expected name
    timeout       it exceeded the timeout and was terminated

None of these is a provider failure: the LLM answered successfully in every
case, and it is the PROGRAM that failed. They map onto the
FailureType.GENERATED_CODE_* categories, which are scorable, and stay
separate from provider_error, which is not.

This is not a security boundary — see the restricted-builtins note on
_build_safe_globals.

Usage in benchmarks:
    python benchmarks/gsm8k_oracle.py --baseline --provider groq --model ...

or directly:
    from benchmarks.baselines.code_gen_baseline import CodeGenBaseline

    baseline = CodeGenBaseline()
    result = baseline.execute(request, backend)
"""

from __future__ import annotations

import ast
import logging
import multiprocessing
import re
from typing import Any, Dict, Optional

from aicomputing.backends.base import Backend
from aicomputing.prompting import build_codegen_prompt
from aicomputing.request import ExecuteRequest
from aicomputing.strategies.base import CallStats, Strategy, StrategyResult

logger = logging.getLogger("aicomputing")


# Restricted builtins for sandboxed execution
_SAFE_BUILTINS = {
    "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod",
    "enumerate", "filter", "float", "format", "frozenset", "hash",
    "hex", "int", "isinstance", "issubclass", "iter", "len", "list",
    "map", "max", "min", "next", "oct", "ord", "pow", "print",
    "range", "repr", "reversed", "round", "set", "slice", "sorted",
    "str", "sum", "tuple", "type", "zip",
    "True", "False", "None",
}


# Modules the generated code may use. Injected as globals AND importable, so
# that both `math.sqrt(x)` and `import math` work. Kept as one list because the
# codegen prompt names it to the model — see prompting.build_codegen_prompt.
_ALLOWED_MODULES = (
    "math", "json", "re", "itertools", "functools",
    "collections", "string", "statistics",
)


def _build_safe_globals() -> Dict[str, Any]:
    """
    Build a restricted globals dict for exec().

    Imports
    -------
    `safe["__builtins__"] = safe` removes __import__ along with everything else
    not whitelisted, which made any `import re` in generated code fail with
    "ImportError: __import__ not found". Models write that line idiomatically,
    and it accounted for 23 of the 30 baseline failures in the first pilot — a
    harness restriction being recorded as a code-generation failure.

    The prompt now tells the model these modules are pre-injected, but a
    restricted __import__ is provided as well rather than relying on the model
    to comply: a baseline that fails whenever the model writes an idiomatic
    import is measuring instruction-following, not code generation.

    This is not a security boundary. It narrows what generated code reaches by
    accident; it does not contain code written to escape. The isolation that
    matters is the separate process with a timeout in _sandbox_exec.
    """
    import builtins
    import importlib

    safe = {k: getattr(builtins, k) for k in _SAFE_BUILTINS if hasattr(builtins, k)}

    modules = {name: importlib.import_module(name) for name in _ALLOWED_MODULES}
    safe.update(modules)

    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root not in modules:
            raise ImportError(
                f"import of {name!r} is not available in this sandbox; "
                f"allowed: {', '.join(_ALLOWED_MODULES)}"
            )
        return modules[root] if not fromlist else importlib.import_module(name)

    safe["__import__"] = _restricted_import
    safe["__builtins__"] = safe
    return safe


def _extract_function_body(raw: str) -> str:
    """Clean up model output to get just the function body."""
    text = raw.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    text = re.split(r"\nif __name__\s*==", text)[0]
    return text


def _assemble_function(body: str, signature: str, docstring: str,
                       func_name: str) -> str:
    """
    Turn the model's output into a module that defines `func_name`.

    Two shapes are accepted because models produce both:

      * a bare body, which is what the prompt asks for, and which is wrapped
        in the declared signature here;
      * a complete `def`, which is returned unchanged.

    Deciding by parsing rather than by looking for a leading "def " matters:
    output that begins with a helper function and defines the target further
    down is still a complete definition, and indenting it would corrupt it.
    Unparseable output falls through to the wrapping path, where the sandbox
    reports the SyntaxError as an ordinary execution failure.
    """
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return _wrap_body(body, signature, docstring)
    defines = any(isinstance(n, ast.FunctionDef) and n.name == func_name
                  for n in tree.body)
    return body if defines else _wrap_body(body, signature, docstring)


def _wrap_body(body: str, signature: str, docstring: str) -> str:
    """Wrap a bare function body in its declared signature and docstring."""
    out = f"def {signature}:\n" + f'    """{docstring}"""\n'
    for line in body.splitlines():
        if line.strip():
            out += (line if line.startswith(("    ", "\t")) else "    " + line) + "\n"
        else:
            out += "\n"
    return out


def _extract_func_name(signature: str) -> str:
    """Extract function name from signature string."""
    m = re.match(r"\s*(\w+)\s*\(", signature)
    return m.group(1) if m else "func"


def _exec_worker(func_code: str, func_name: str, inputs: Dict[str, Any], conn) -> None:
    """
    Child-process entry point: exec the generated function and send back
    its return value. Must be module-level so it survives pickling under
    the spawn start method (Windows).
    """
    try:
        safe_globals = _build_safe_globals()
        local_ns: Dict[str, Any] = {}
        compiled = compile(func_code, filename="<codegen_baseline>", mode="exec")
        exec(compiled, safe_globals, local_ns)
        if func_name not in local_ns:
            conn.send(("error", f"Generated code did not define function {func_name!r}"))
            return
        conn.send(("ok", local_ns[func_name](**inputs)))
    except BaseException as exc:  # includes the unpicklable-result case below
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            conn.send(("error", "generated code raised an unreportable error"))
    finally:
        conn.close()


class CodeGenBaseline(Strategy):
    """
    Comparison baseline: generate code, execute it, return the real output.

    This is NOT AI computing. It is the traditional LLM-for-code approach.
    Included for benchmark comparison only.
    """

    def __init__(self, timeout: float = 10.0) -> None:
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
        prompt = build_codegen_prompt(
            signature=request.signature,
            docstring=request.docstring,
        )

        response = backend.generate(
            prompt,
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=max(request.max_new_tokens or 0, 512),
        )

        # One generation call; recorded so the baseline appears in the cost
        # comparison on the same footing as the spec-only arms.
        stats = CallStats().record(response)

        body = _extract_function_body(response.text)
        func_name = _extract_func_name(request.signature)

        # Reconstruct the full function.
        #
        # The prompt asks for a body, but models frequently return a complete
        # `def` instead. Wrapping one of those in another `def` produced a
        # nested definition whose body sat at the same indentation as the inner
        # header - an IndentationError scored as a code-generation failure when
        # the generated code was in fact fine. That was 10 of 30 baseline
        # failures on the first pilot. If the model already defined the target
        # function, its code is used as written.
        func_code = _assemble_function(body, request.signature,
                                       request.docstring, func_name)

        # Execute in sandbox.
        #
        # SCORING RULE (thesis §5.5): a generation that does not execute scores
        # as WRONG. There is deliberately no fallback to parsing the generated
        # source text for an answer.
        #
        # The previous behaviour did exactly that, and it is not a conservative
        # choice — it is a contaminating one. This baseline's whole claim is
        # "generate code, run it, return the real output". When the code does
        # not run, reading a value out of the source text instead means the
        # baseline silently borrows the spec-only mechanism it exists to be
        # compared against, so §6.3 would be measuring the gap between
        # spec-only prediction and a mixture that already contains it. The
        # measured gap would be understated and the baseline's accuracy
        # inflated, in an amount that varies with how often generation fails.
        #
        # The failure is recorded rather than hidden: exec_status carries the
        # category so the execution-failure rate is reportable alongside
        # accuracy instead of being folded into it.
        # Compile separately from running so a program that does not parse is
        # distinguishable from one that parses and then raises. Both are the
        # model's fault, but they say different things about what generation
        # got wrong, and neither is a provider failure.
        exec_status = "ok"
        try:
            compile(func_code, filename="<codegen_baseline>", mode="exec")
        except SyntaxError:
            exec_status = "syntax_error"
            value = None
            logger.info("[codegen] generated code does not parse; scored as wrong")
        if exec_status == "ok":
            try:
                value = self._sandbox_exec(func_code, func_name, request.inputs)
            except TimeoutError:
                value, exec_status = None, "timeout"
                logger.info("[codegen] generated code timed out; scored as wrong")
            except Exception as exc:
                value, exec_status = None, "runtime_error"
                logger.info("[codegen] generated code raised (%s); scored as wrong",
                            type(exc).__name__)

        return StrategyResult(
            value=value,
            prompts=[prompt],
            raw_text=f"[codegen {exec_status}] {response.text}",
            tokens_used=response.tokens_used,
            provider_finish_reason=response.provider_finish_reason,
            exec_status=exec_status,
            call_stats=stats,
        )

    def _sandbox_exec(
        self,
        func_code: str,
        func_name: str,
        inputs: Dict[str, Any],
    ) -> Any:
        """
        Execute generated code in a restricted environment, in a separate
        process so `self._timeout` can actually be enforced.

        A generated infinite loop cannot be interrupted in-process (signal
        alarms are POSIX-only and threads are not killable), so the child
        is terminated on timeout and TimeoutError is raised.
        """
        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
        proc = multiprocessing.Process(
            target=_exec_worker,
            args=(func_code, func_name, inputs, child_conn),
        )
        proc.start()
        child_conn.close()  # only the child keeps the sending end open

        try:
            if not parent_conn.poll(self._timeout):
                raise TimeoutError(
                    f"Generated code exceeded the {self._timeout}s timeout"
                )
            status, payload = parent_conn.recv()
        except EOFError as exc:
            raise RuntimeError("Generated code died without returning a result") from exc
        finally:
            parent_conn.close()
            if proc.is_alive():
                proc.terminate()
            proc.join()

        if status == "error":
            raise RuntimeError(payload)
        return payload
