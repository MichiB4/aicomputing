"""
CLI entry point for AI Computing.

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
    argv = argv or sys.argv[1:]
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
    p.add_argument("--base-url", default=None, help="API base URL (for openai backend)")
    p.add_argument("--api-key", default=None, help="API key (for openai backend)")
    p.add_argument("--strategy", default="direct", choices=list_strategies())
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=None, dest="max_tokens",
               help="Override the starting token budget (default: auto per model)")
    p.add_argument("--no-strict", action="store_true", help="Disable strict type checking")
    p.add_argument("--trace", default=None, help="Write JSONL trace of LLM calls")
    think_group = p.add_mutually_exclusive_group()
    think_group.add_argument("--think", action="store_true", default=None, help="Force thinking ON")
    think_group.add_argument("--no-think", action="store_true", dest="no_think", help="Force thinking OFF")
    p.add_argument("--verbose", "-v", action="store_true")

    args = p.parse_args(argv)

    think_override = None
    if args.think:
        think_override = True
    elif args.no_think:
        think_override = False

    type_map = {
        None: None, "int": int, "float": float,
        "str": str, "bool": bool, "none": type(None),
    }
    return_type = type_map[args.return_type]

    backend_kwargs = {}
    if args.base_url:
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
        think=think_override,
    )

    res = vr.execute(req)

    # Output
    print(json.dumps(res.result, ensure_ascii=False, default=str))

    if args.verbose:
        print(f"[strategy] {res.strategy}", file=sys.stderr)
        print(f"[model]    {res.model}", file=sys.stderr)
        print(f"[latency]  {res.latency_ms:.0f}ms", file=sys.stderr)
        print(f"[tokens]   {res.tokens_used}", file=sys.stderr)
        print(f"[retries]  {res.retries}", file=sys.stderr)
        if res.type_warnings:
            print(f"[type_warn] {res.type_warnings}", file=sys.stderr)
        print(f"[raw] {res.raw_text}", file=sys.stderr)


if __name__ == "__main__":
    main()
