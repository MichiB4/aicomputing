"""
Shared argument parsing for AI Computing benchmark scripts.

Adds a --provider flag that auto-configures base_url for known providers,
and auto-derives sensible think/strategy defaults from the model name using
the model registry.

Usage in a benchmark script::

    from bench_args import add_common_args, resolve_args

    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--limit", type=int, default=None)   # benchmark-specific
    args = ap.parse_args()
    cfg = resolve_args(args)

    vr = VirtualRuntime(
        backend=cfg.backend,
        model=cfg.model,
        strategy=cfg.strategy,
        **cfg.backend_kwargs,
    )
    # cfg.think_override  →  None | True | False
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Add the src directory to the path so aicomputing is importable when running
# benchmarks directly (without pip install -e .)
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aicomputing.model_config import (
    ThinkingMode,
    get_thinking_mode,
    get_provider_preset,
    list_providers,
    PROVIDER_PRESETS,
)


@dataclass
class BenchConfig:
    backend: str
    model: str
    strategy: str
    think_override: Optional[bool]
    backend_kwargs: Dict[str, Any] = field(default_factory=dict)
    trace: Optional[str] = None


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """
    Add the standard set of backend / model / strategy arguments to a parser.

    --provider  Short name for a known cloud provider (groq, cerebras,
                openrouter, sambanova, ollama). Auto-fills --base-url and
                sets --backend openai. Overridden by explicit --base-url.

    --backend   Backend name (ollama, openai). Default: ollama.
    --model     Model name. Default depends on provider.
    --base-url  Override the API base URL.
    --api-key   API key (or set via environment variable).
    --strategy  direct | cot. Auto-detected from model if omitted.
    --think / --no-think  Override thinking mode. Auto-detected if omitted.
    --trace     Path to write a JSONL trace file.
    """
    provider_choices = list_providers()
    ap.add_argument(
        "--provider",
        default=None,
        choices=provider_choices,
        help=(
            f"Known provider shortcut: {', '.join(provider_choices)}. "
            "Auto-sets --base-url and --backend. "
            "You still need to supply --api-key and --model."
        ),
    )
    ap.add_argument("--backend", default=None,
                    help="Backend name (ollama, openai). Auto-set by --provider.")
    ap.add_argument("--model", default=None,
                    help="Model name. Required unless using --provider with a single model.")
    ap.add_argument("--base-url", default=None,
                    help="API base URL. Auto-set by --provider.")
    ap.add_argument("--api-key", default=None,
                    help="API key for cloud providers.")
    ap.add_argument(
        "--strategy",
        default=None,
        choices=["direct", "cot"],
        help=(
            "Prompting strategy. "
            "Default: 'cot' for thinking models, 'direct' for non-thinking models."
        ),
    )
    ap.add_argument("--trace", default=None,
                    help="Path to write a JSONL trace file.")
    think_group = ap.add_mutually_exclusive_group()
    think_group.add_argument(
        "--think", action="store_true", default=None,
        help="Force thinking mode ON.",
    )
    think_group.add_argument(
        "--no-think", action="store_true", dest="no_think",
        help="Force thinking mode OFF.",
    )


def resolve_args(args: argparse.Namespace) -> BenchConfig:
    """
    Resolve provider presets, auto-detect strategy and think defaults,
    and return a BenchConfig ready to pass to VirtualRuntime.
    """
    backend = args.backend
    model = args.model
    base_url = args.base_url
    api_key = args.api_key

    # --- 1. Apply provider preset ---
    if args.provider:
        preset = get_provider_preset(args.provider)
        if preset:
            if not base_url:
                base_url = preset["base_url"]
            if not backend:
                backend = "ollama" if args.provider == "ollama" else "openai"
            if not model and len(preset["models"]) == 1:
                model = preset["models"][0]
                print(f"[bench_args] --provider {args.provider}: auto-selected model '{model}'")
            # Print available models for this provider if model not set
            if not model:
                print(
                    f"[bench_args] --provider {args.provider} available models: "
                    + ", ".join(preset["models"])
                )
                print("[bench_args] Please specify --model explicitly.")
                sys.exit(1)

    # --- 2. Final defaults ---
    if not backend:
        backend = "ollama"
    if not model:
        model = "qwen3:4b-instruct"
        print(f"[bench_args] No model specified — using default '{model}'")

    # --- 3. Auto-detect think default from model ---
    thinking_mode = get_thinking_mode(model)

    if args.think:
        think_override = True
    elif args.no_think:
        think_override = False
    else:
        # Default: think ON for thinking models, OFF for non-thinking
        if thinking_mode == ThinkingMode.NO_THINKING:
            think_override = False
        else:
            think_override = True  # ALWAYS_THINKS or THINKING_TOGGLE

    # --- 4. Auto-detect strategy from model and think setting ---
    strategy = args.strategy
    if strategy is None:
        if thinking_mode == ThinkingMode.NO_THINKING or not think_override:
            strategy = "direct"
        else:
            strategy = "cot"

    # --- 5. Build backend kwargs ---
    backend_kwargs: Dict[str, Any] = {}
    if base_url:
        if backend == "ollama":
            # OllamaBackend uses 'host' and the Ollama Python client does not
            # want the /v1 suffix — strip it if present.
            host = base_url.rstrip("/")
            if host.endswith("/v1"):
                host = host[:-3]
            # Only set host if it's not the default (avoids redundant arg)
            if host not in ("http://localhost:11434", "http://127.0.0.1:11434"):
                backend_kwargs["host"] = host
        else:
            backend_kwargs["base_url"] = base_url
    if api_key:
        backend_kwargs["api_key"] = api_key

    # --- 6. For Ollama, verify the model is pulled ---
    if backend == "ollama":
        try:
            import ollama as _ollama
            pulled = [m.model for m in _ollama.list().models]
            # Normalise: ollama list may return "qwen3:4b-instruct" or "qwen3:4b-instruct:latest"
            pulled_base = [m.split(":")[0] + ":" + m.split(":")[1] if m.count(":") >= 1 else m
                           for m in pulled]
            model_base = model.split(":latest")[0]
            if model not in pulled and model_base not in pulled and model not in pulled_base:
                print(f"[bench_args] WARNING: model '{model}' not found in local Ollama.")
                print(f"[bench_args] Available models: {pulled}")
                print(f"[bench_args] Run: ollama pull {model}")
                print()
        except Exception:
            pass  # Ollama not running — will fail at first generate() call with a clearer error
    thinking_label = {
        ThinkingMode.ALWAYS_THINKS: "always on (DeepSeek-R1 style)",
        ThinkingMode.THINKING_TOGGLE: f"{'ON' if think_override else 'OFF'} (toggleable)",
        ThinkingMode.NO_THINKING: "not supported (instruct model)",
    }[thinking_mode]

    print(f"Provider: {args.provider or 'custom'}")
    print(f"Backend:  {backend}")
    print(f"Model:    {model}")
    print(f"Strategy: {strategy}")
    print(f"Thinking: {thinking_label}")
    if base_url:
        print(f"Base URL: {base_url}")
    print()

    return BenchConfig(
        backend=backend,
        model=model,
        strategy=strategy,
        think_override=think_override,
        backend_kwargs=backend_kwargs,
        trace=args.trace,
    )


def print_provider_table() -> None:
    """Print a table of available providers and their models."""
    print("Available providers (--provider):")
    print()
    for name, preset in PROVIDER_PRESETS.items():
        print(f"  {name:<12} {preset['notes']}")
        for m in preset["models"]:
            print(f"               • {m}")
    print()
