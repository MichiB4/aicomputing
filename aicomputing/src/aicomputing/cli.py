"""
CLI entry point for AI Computing.

Prompting strategies and native reasoning settings are independent. Native
options are omitted by default; recorded requests do not prove model compliance.

Results print as JSON when JSON preserves the value exactly, otherwise as a
Python representation (for example sets, tuples, bytes or non-string keys).
Failed predictions leave stdout empty, report an error on stderr, and exit 1.
WARN retains invalid values with a warning; a valid None remains a success.

Usage:
    aicomp "sort(arr: list[int]) -> list[int]" \
        "Return a sorted copy of the array." \
        --inputs '{"arr": [5, 3, 1]}' \
        --backend ollama --model qwen3:4b --strategy direct
"""

from __future__ import annotations

import argparse
import json
import sys

from .request import ExecuteRequest
from .runtime import VirtualRuntime
from .status import ValidationStatus
from .backends.base import BackendCommunicationError, ToolUseDetected
from .strategies import CodeExecutionError, create_strategy, list_strategies
from .types import parse_signature
from .validation import validate_inputs, validate_nonblank_text


def main(argv=None):
    p = argparse.ArgumentParser(
        description="AI Computing: predict function outputs from specs",
    )
    p.add_argument("signature", help="Function signature, e.g. 'add(x: int, y: int) -> int'")
    p.add_argument("docstring", help="Natural language description of the function")
    p.add_argument("--inputs", required=True, help="JSON dict of concrete inputs")
    p.add_argument("--backend", required=True, choices=["ollama", "hf", "openai"],
                   help="Inference backend; install its corresponding optional extra")
    p.add_argument("--model", default=None, help="Model name")
    p.add_argument("--base-url", default=None, help="OpenAI API base URL or Ollama host, passed unchanged")
    p.add_argument("--api-key", default=None, help="API key (for openai backend)")
    p.add_argument("--strategy", default="direct", choices=list_strategies())
    p.add_argument("--execution-timeout", type=float, default=120.0,
                   help="code_execute worker timeout per attempt in seconds (default: 120)")
    p.add_argument("--execution-retries", type=int, default=2,
                   help="code_execute additional local timeout attempts (default: 2)")
    p.add_argument("--code-task-mode", choices=["function", "word_problem"], default="function",
                   help="code_execute: implement a general function or arithmetic for one supplied problem")
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=None, dest="max_tokens",
               help="Fixed output-token limit per call; omitted means no application cap")
    p.add_argument("--validation-mode", choices=["strict", "warn", "none"], default="strict",
                   help="Output checks: withhold invalid values, report warnings, or skip checks (default: strict)")
    p.add_argument("--trace", default=None, help="Write JSONL trace of LLM calls")
    p.add_argument("--reasoning-effort", default=None,
                   help="OpenAI-compatible native effort string; omitted by default")
    p.add_argument("--ollama-think", default=None, metavar="VALUE",
                   help="Ollama true/false or a model-specific level; omitted by default")
    p.add_argument("--hf-thinking", action=argparse.BooleanOptionalAction, default=None,
                   help="HF chat-template enable_thinking override; omitted by default")
    p.add_argument("--verbose", "-v", action="store_true")

    args = p.parse_args(argv)
    try:
        inputs = json.loads(args.inputs)
    except ValueError as exc:
        p.error(f"--inputs is not valid JSON: {exc}")

    native_options = (
        ("--reasoning-effort", args.reasoning_effort, "openai", "reasoning_effort"),
        ("--ollama-think", args.ollama_think, "ollama", "think"),
        ("--hf-thinking/--no-hf-thinking", args.hf_thinking, "hf", "enable_thinking"),
    )
    backend_kwargs = {}
    for option, value, backend, key in native_options:
        if value is not None:
            if args.backend != backend:
                p.error(f"{option} requires --backend {backend}")
            if isinstance(value, str):
                try:
                    validate_nonblank_text(value, option)
                except (TypeError, ValueError) as exc:
                    p.error(str(exc))
            if key == "think":
                value = {"true": True, "false": False}.get(value, value)
            backend_kwargs[key] = value

    if args.backend == "hf" and (args.base_url is not None or args.api_key is not None):
        p.error("Backend 'hf' does not accept an API URL or API key")
    if args.backend == "ollama" and args.api_key is not None:
        p.error("This Ollama adapter does not accept an API key")
    try:
        validate_nonblank_text(args.base_url, "--base-url", optional=True)
        validate_nonblank_text(args.api_key, "--api-key", optional=True)
        validate_nonblank_text(args.trace, "--trace", optional=True)
    except (TypeError, ValueError) as exc:
        p.error(str(exc))
    if args.base_url is not None:
        if args.backend == "ollama":
            backend_kwargs["host"] = args.base_url
        else:
            backend_kwargs["base_url"] = args.base_url
    if args.api_key is not None:
        backend_kwargs["api_key"] = args.api_key

    strategy = args.strategy
    if strategy == "code_execute":
        try:
            strategy = create_strategy(
                strategy, execution_timeout=args.execution_timeout,
                execution_retries=args.execution_retries, task_mode=args.code_task_mode,
            )
        except (TypeError, ValueError) as exc:
            p.error(str(exc))

    try:
        req = ExecuteRequest(
            signature=args.signature,
            docstring=args.docstring,
            inputs=inputs,
            model=args.model,
            strategy=args.strategy,
            temperature=args.temp,
            max_new_tokens=args.max_tokens,
        )
        validate_inputs(parse_signature(req.signature), req.inputs)
    except (TypeError, ValueError) as exc:
        p.error(str(exc))

    try:
        with VirtualRuntime(
            backend=args.backend,
            model=args.model,
            strategy=strategy,
            validation_mode=args.validation_mode,
            trace_file=args.trace,
            **backend_kwargs,
        ) as vr:
            res = vr.execute(req)
    except CodeExecutionError as exc:
        _diagnostic("error", str(exc))
        return 1
    except ModuleNotFoundError as exc:
        _diagnostic("error", f"{exc}. Install the backend dependencies with: pip install 'aicomputing[{args.backend}]'.")
        return 1
    except (BackendCommunicationError, ToolUseDetected, ImportError) as exc:
        _diagnostic("error", str(exc))
        return 1

    warning = args.validation_mode == "warn" and res.validation_status in (ValidationStatus.INVALID_TYPE, ValidationStatus.INVALID_CONSTRAINT)
    failed = not warning and (res.failure_type is not None or res.validation_status not in (ValidationStatus.VALID, ValidationStatus.NOT_CHECKED))
    if failed or warning:
        detail = res.execution_error or res.parse_error or "; ".join(res.type_warnings + res.constraint_warnings)
        message = res.validation_status + (": " + detail if detail else "")
        _diagnostic("error" if failed else "warning", message)
    if not failed:
        print(_format_result(res.result))

    if args.verbose:
        print(f"[strategy] {res.strategy}", file=sys.stderr)
        print(f"[model]    {res.model}", file=sys.stderr)
        print(f"[latency]  {res.latency_ms:.0f}ms", file=sys.stderr)
        print(f"[input_tokens] {res.input_tokens}", file=sys.stderr)
        print(f"[output_tokens] {res.output_tokens}", file=sys.stderr)
        print(f"[total_tokens] {res.total_tokens}", file=sys.stderr)
        print(f"[validation_retries] {res.retries}", file=sys.stderr)
        print(f"[validation] {res.validation_status}", file=sys.stderr)
        print(f"[reasoning_requests] {res.reasoning_requests}", file=sys.stderr)
        if res.type_warnings:
            print(f"[type_warn] {res.type_warnings}", file=sys.stderr)
        if res.constraint_warnings:
            print(f"[constraint_warn] {res.constraint_warnings}", file=sys.stderr)
        if res.parse_error:
            print(f"[parse_error] {res.parse_error}", file=sys.stderr)
        if res.execution_error:
            print(f"[execution_error] {res.execution_error}", file=sys.stderr)
        print(f"[raw] {res.raw_text}", file=sys.stderr)
    return 1 if failed else 0


def _diagnostic(level, message):
    """Keep ordinary errors and warnings concise; verbose output retains details."""
    print(f"aicomp: {level}: " + " ".join(message.split())[:500], file=sys.stderr)


def _format_result(value) -> str:
    """Use JSON only when decoding it gives back the same value and types."""
    try:
        text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)
    if _same_json_value(json.loads(text), value):
        return text
    return repr(value)


def _same_json_value(decoded, value) -> bool:
    if type(decoded) is not type(value):
        return False
    if isinstance(value, list):
        return len(decoded) == len(value) and all(map(_same_json_value, decoded, value))
    if isinstance(value, dict):
        return list(decoded) == list(value) and all(
            _same_json_value(decoded[key], value[key]) for key in value
        )
    return decoded == value


if __name__ == "__main__":
    raise SystemExit(main())
