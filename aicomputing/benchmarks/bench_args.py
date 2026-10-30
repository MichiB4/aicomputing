"""Shared benchmark CLI arguments, connection presets, and runtime construction.

Prompting defaults to direct, independently of native reasoning. Native overrides
apply only to their backend; omission leaves provider defaults unchanged.
Requested settings do not prove their effect.

Call configure_system_certificates() at application startup before HTTPS imports.
build_runtime() applies tracing and the selected oracle strategy."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Add the src directory to the path so aicomputing is importable when running
# benchmarks directly (without pip install -e .)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aicomputing import VirtualRuntime
from aicomputing.model_config import (
    get_provider_preset,
    list_providers,
)
from aicomputing.backends.base import BackendCommunicationError, BackendTimeoutError
from aicomputing.request import ExecuteResult, FailureType
from aicomputing.strategies import list_strategies


@dataclass
class BenchConfig:
    backend: str
    model: str
    strategy: str
    backend_kwargs: Dict[str, Any] = field(default_factory=dict)
    trace: Optional[str] = None
    max_tokens: Optional[int] = None


def configure_system_certificates() -> None:
    """Configure optional system certificates during application startup."""
    try:
        import truststore
    except ModuleNotFoundError as exc:
        if exc.name != "truststore":
            raise
        return
    truststore.inject_into_ssl()


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """Register provider, connection, strategy, generation, and native-control options.

    Reasoning controls are omitted by default and rejected for other backends."""
    provider_choices = list_providers()
    ap.add_argument(
        "--provider",
        default=None,
        choices=provider_choices,
        help=(
            f"Known provider shortcut: {', '.join(provider_choices)}. "
            "Supplies defaults for --base-url and --backend. "
            "Single-model presets can supply the model; "
            "provider environment variables can supply credentials."
        ),
    )
    ap.add_argument("--backend", default=None,
                    help="Backend name (ollama, openai, hf). Defaults to provider backend or ollama.")
    ap.add_argument("--model", default=None,
                    help="Model identifier; presets or fallback may supply a default.")
    ap.add_argument("--base-url", default=None,
                    help="API base URL. Auto-set by --provider.")
    ap.add_argument("--api-key", default=None,
                    help="API key for cloud providers.")
    ap.add_argument(
        "--strategy",
        default=None,
        choices=list_strategies(),
        help=(
            "Prompting strategy. "
            "Default: 'direct', independent of native reasoning settings."
        ),
    )
    ap.add_argument("--trace", default=None,
                    help="Path to write a JSONL trace file.")
    ap.add_argument(
        "--max-tokens", type=int, default=None, dest="max_tokens",
        help=(
            "Optional starting output-token allowance; truncation retries may increase it. "
            "When omitted, inherit the manifest setting, or use no application cap."
        ),
    )
    ap.add_argument(
        "--reasoning-effort", default=None,
        help="OpenAI-compatible reasoning_effort value; omitted by default.",
    )
    ap.add_argument(
        "--ollama-think", default=None, metavar="VALUE",
        help=(
            "Ollama think value: true, false, or a model-specific level. "
            "Omitted by default."
        ),
    )
    ap.add_argument(
        "--hf-thinking", action=argparse.BooleanOptionalAction, default=None,
        help="HF chat-template enable_thinking override; omitted by default.",
    )


def resolve_args(args: argparse.Namespace) -> BenchConfig:
    """Resolve presets and CLI options; default to direct without inferring reasoning."""
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")

    backend = args.backend
    model = args.model
    base_url = args.base_url
    api_key = args.api_key

    # Use the provider's environment key when no explicit key was supplied.
    if not api_key and args.provider:
        api_key = os.environ.get(_PROVIDER_ENV_KEYS.get(args.provider.lower(), ""))

    # 1. Apply provider preset
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

    # 2. Final defaults
    if not backend:
        backend = "ollama"
    if not model:
        if backend != "ollama":
            raise ValueError(f"--model is required for backend '{backend}'")
        model = "qwen3:4b-instruct"
        print(f"[bench_args] No model specified — using default '{model}'")

    # 3. Select the prompting strategy independently of native reasoning
    strategy = args.strategy or "direct"

    # Reject connection settings unsupported by the selected adapter.
    if backend == "hf" and (base_url or api_key):
        raise ValueError("Backend 'hf' does not accept an API URL or API key")
    if backend == "ollama" and api_key:
        raise ValueError("This Ollama adapter does not accept an API key")

    # 4. Build backend kwargs
    backend_kwargs: Dict[str, Any] = {}
    if base_url:
        if backend == "ollama":
            # Pass the normalized address even when it points to localhost.
            host = base_url.rstrip("/")
            if host.endswith("/v1"):
                host = host[:-3]
            backend_kwargs["host"] = host
        else:
            backend_kwargs["base_url"] = base_url
    if api_key:
        backend_kwargs["api_key"] = api_key

    # Native controls belong to their backend, independently of strategy.
    if args.reasoning_effort is not None:
        if backend != "openai":
            raise ValueError("--reasoning-effort requires --backend openai")
        backend_kwargs["reasoning_effort"] = args.reasoning_effort

    if args.ollama_think is not None:
        if backend != "ollama":
            raise ValueError("--ollama-think requires --backend ollama")
        value = args.ollama_think
        backend_kwargs["think"] = {"true": True, "false": False}.get(value, value)

    if args.hf_thinking is not None:
        if backend != "hf":
            raise ValueError("--hf-thinking/--no-hf-thinking requires --backend hf")
        backend_kwargs["enable_thinking"] = args.hf_thinking

    reasoning_overrides = {
        key: backend_kwargs[key]
        for key in ("reasoning_effort", "think", "enable_thinking")
        if key in backend_kwargs
    }

    print(f"Provider: {args.provider or 'custom'}")
    print(f"Backend:  {backend}")
    print(f"Model:    {model}")
    print(f"Strategy: {strategy}")
    print(f"Reasoning overrides requested: {reasoning_overrides}")
    if base_url:
        print(f"Base URL: {base_url}")
    print()

    return BenchConfig(
        backend=backend,
        model=model,
        strategy=strategy,
        backend_kwargs=backend_kwargs,
        trace=args.trace,
        max_tokens=args.max_tokens,
    )


# Conventional environment variable per provider. Ollama needs none.
_PROVIDER_ENV_KEYS = {
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
}

def build_runtime(cfg: BenchConfig):
    """Build the oracle runtime; token allowances belong to execution requests."""

    return VirtualRuntime(
        backend=cfg.backend,
        model=cfg.model,
        strategy=cfg.strategy,
        trace_file=cfg.trace,
        **cfg.backend_kwargs,
    )


def execute_resilient(vr, req, *, label: str = "") -> Any:
    """Convert invocation failures into excluded benchmark rows, without retrying.

    Explicit backend communication errors are provider failures, and backend
    timeout errors are timeouts. Other invocation errors are harness failures.
    KeyboardInterrupt, SystemExit and GeneratorExit propagate.
    """
    started = time.perf_counter()
    try:
        return vr.execute(req)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        print(f"    [example failed] {label or ''} {detail}".rstrip(), flush=True)
        if isinstance(exc, BackendTimeoutError):
            failure_type = FailureType.TIMEOUT
        elif isinstance(exc, BackendCommunicationError):
            failure_type = FailureType.PROVIDER_ERROR
        else:
            failure_type = FailureType.HARNESS_ERROR

        return ExecuteResult(
            result=None,
            raw_text=f"[execute failed] {detail}",
            model=getattr(vr, "model", ""),
            strategy=getattr(getattr(vr, "strategy", None), "name", ""),
            failure_type=failure_type,
            scorable=False,
            validation_status="unparsed",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            tokens_used=0,
            token_retries=0,
            answer_format=getattr(req, "answer_format", "legacy"),
            transport_diagnostics=[exc.diagnostics] if getattr(exc, "diagnostics", None) is not None else [],
        )
