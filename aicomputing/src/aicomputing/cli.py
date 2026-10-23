"""
CLI entry point for AI Computing.

Prompting strategies and native reasoning settings are independent. Native
options are omitted by default; recorded requests do not prove model compliance.

Results print as JSON when possible, with a Python representation as fallback.

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
from .strategies import list_strategies


def main(argv=None):
    p = argparse.ArgumentParser(
        description="AI Computing: predict function outputs from specs",
    )
    p.add_argument("signature", help="Function signature, e.g. 'add(x: int, y: int) -> int'")
    p.add_argument("docstring", help="Natural language description of the function")
    p.add_argument("--inputs", required=True, help="JSON dict of concrete inputs")
    p.add_argument("--return-type", default=None,
                   choices=["int", "float", "str", "bool", "none"],
                   help="Return type for coercion")
    p.add_argument("--backend", default="ollama", choices=["ollama", "hf", "openai"])
    p.add_argument("--model", default=None, help="Model name")
    p.add_argument("--base-url", default=None, help="OpenAI API base URL or Ollama host")
    p.add_argument("--api-key", default=None, help="API key (for openai backend)")
    p.add_argument("--strategy", default="direct", choices=list_strategies())
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=None, dest="max_tokens",
               help="Starting token budget; automatic truncation retries may increase it")
    p.add_argument("--no-strict", action="store_true", help="Disable strict type checking")
    p.add_argument("--trace", default=None, help="Write JSONL trace of LLM calls")
    p.add_argument("--reasoning-effort", default=None,
                   help="OpenAI-compatible native effort string; omitted by default")
    p.add_argument("--ollama-think", default=None, metavar="VALUE",
                   help="Ollama true/false or a model-specific level; omitted by default")
    p.add_argument("--hf-thinking", action=argparse.BooleanOptionalAction, default=None,
                   help="HF chat-template enable_thinking override; omitted by default")
    p.add_argument("--verbose", "-v", action="store_true")

    args = p.parse_args(argv)

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
            if key == "think":
                value = {"true": True, "false": False}.get(value, value)
            backend_kwargs[key] = value

    type_map = {
        None: None, "int": int, "float": float,
        "str": str, "bool": bool, "none": type(None),
    }
    return_type = type_map[args.return_type]

    if args.backend == "hf" and (args.base_url or args.api_key):
        p.error("Backend 'hf' does not accept an API URL or API key")
    if args.backend == "ollama" and args.api_key:
        p.error("This Ollama adapter does not accept an API key")
    if args.base_url:
        if args.backend == "ollama":
            host = args.base_url.rstrip("/")
            if host.endswith("/v1"):
                host = host[:-3]
            backend_kwargs["host"] = host
        else:
            backend_kwargs["base_url"] = args.base_url
    if args.api_key:
        backend_kwargs["api_key"] = args.api_key

    vr = VirtualRuntime(
        backend=args.backend,
        model=args.model,
        strategy=args.strategy,
        strict_types=not args.no_strict,
        trace_file=args.trace,
        **backend_kwargs,
    )

    req = ExecuteRequest(
        signature=args.signature,
        docstring=args.docstring,
        inputs=json.loads(args.inputs),
        return_type=return_type,
        temperature=args.temp,
        max_new_tokens=args.max_tokens,
        strict_types=not args.no_strict,
    )

    res = vr.execute(req)

    # Output
    try:
        output = json.dumps(res.result, ensure_ascii=False, default=str)
    except TypeError:
        output = repr(res.result)
    print(output)

    if args.verbose:
        print(f"[strategy] {res.strategy}", file=sys.stderr)
        print(f"[model]    {res.model}", file=sys.stderr)
        print(f"[latency]  {res.latency_ms:.0f}ms", file=sys.stderr)
        print(f"[tokens]   {res.tokens_used}", file=sys.stderr)
        print(f"[retries]  {res.retries}", file=sys.stderr)
        print(f"[reasoning_requests] {res.reasoning_requests}", file=sys.stderr)
        if res.type_warnings:
            print(f"[type_warn] {res.type_warnings}", file=sys.stderr)
        print(f"[raw] {res.raw_text}", file=sys.stderr)


if __name__ == "__main__":
    main()
