"""
Code generation + execution baseline.

THIS IS NOT AN AI COMPUTING STRATEGY. It represents the *traditional*
approach: generate code from a spec, execute it, return the real output.
It lives in benchmarks/baselines/ because it exists solely to compare
against true AI computing (spec-only prediction).

The comparison answers the question: "When does predicting the output
from a spec alone match or beat generating-and-running actual code?"

Usage in benchmarks:
    from benchmarks.baselines.code_gen_baseline import CodeGenBaseline

    baseline = CodeGenBaseline()
    result = baseline.execute(request, backend)
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from aicomputing.backends.base import Backend
from aicomputing.parsing import parse_result
from aicomputing.prompting import build_codegen_prompt
from aicomputing.request import ExecuteRequest
from aicomputing.strategies.base import Strategy, StrategyResult


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


def _build_safe_globals() -> Dict[str, Any]:
    """Build a restricted globals dict for exec()."""
    import builtins
    safe = {k: getattr(builtins, k) for k in _SAFE_BUILTINS if hasattr(builtins, k)}
    safe["__builtins__"] = safe
    import math, json, re as _re, itertools, functools, collections, string, statistics
    safe.update({
        "math": math, "json": json, "re": _re,
        "itertools": itertools, "functools": functools,
        "collections": collections, "string": string,
        "statistics": statistics,
    })
    return safe


def _extract_function_body(raw: str) -> str:
    """Clean up model output to get just the function body."""
    text = raw.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    text = re.split(r"\nif __name__\s*==", text)[0]
    return text


def _extract_func_name(signature: str) -> str:
    """Extract function name from signature string."""
    m = re.match(r"\s*(\w+)\s*\(", signature)
    return m.group(1) if m else "func"


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

        body = _extract_function_body(response.text)
        func_name = _extract_func_name(request.signature)

        # Reconstruct the full function
        func_code = f"def {request.signature}:\n"
        func_code += f'    """{request.docstring}"""\n'
        for line in body.splitlines():
            if line.strip():
                if line.startswith("    ") or line.startswith("\t"):
                    func_code += line + "\n"
                else:
                    func_code += "    " + line + "\n"
            else:
                func_code += "\n"

        # Execute in sandbox
        try:
            value = self._sandbox_exec(func_code, func_name, request.inputs)
        except Exception:
            # If execution fails, fall back to parsing the raw text
            value = parse_result(response.text, return_type=request.return_type)

        return StrategyResult(
            value=value,
            raw_text=response.text,
            tokens_used=response.tokens_used,
        )

    def _sandbox_exec(
        self,
        func_code: str,
        func_name: str,
        inputs: Dict[str, Any],
    ) -> Any:
        """Execute generated code in a restricted environment."""
        safe_globals = _build_safe_globals()
        local_ns: Dict[str, Any] = {}

        compiled = compile(func_code, filename="<codegen_baseline>", mode="exec")
        exec(compiled, safe_globals, local_ns)

        if func_name not in local_ns:
            raise RuntimeError(f"Generated code did not define function {func_name!r}")

        return local_ns[func_name](**inputs)
