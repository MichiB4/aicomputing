"""
Shared argument parsing and runtime construction for benchmark scripts.

Provider presets supply default connection settings. The prompting strategy
defaults to direct, independently of the model name and native reasoning.

Native reasoning overrides are explicit backend settings. Omitting them leaves
backend defaults unchanged; requesting them does not prove the model uses them.

Usage in a benchmark script::

    import argparse
    from bench_args import (
        add_common_args, build_runtime, configure_system_certificates, resolve_args,
    )

    # In the application entry point, before importing HTTPS clients:
    configure_system_certificates()

    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--limit", type=int, default=None)
    cfg = resolve_args(ap.parse_args())
    vr = build_runtime(cfg)

The shared builder applies trace configuration and the optional code-generation
baseline. Native reasoning settings reach the backend through cfg.backend_kwargs.
"""

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
    PROVIDER_PRESETS,
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
    # True when --baseline was passed: run the code-generation baseline
    # instead of a spec-only strategy. Kept as a separate flag rather than a
    # registry entry because the baseline generates and executes code, which
    # every strategy is forbidden to do — see build_runtime().
    baseline: bool = False


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
    """Register shared benchmark command-line options.

    --provider       Supply default connection settings from a named preset.
                     Explicit backend and URL options take precedence.
    --backend        Backend name; defaults to ollama without a provider.
    --model          Model identifier, resolved from the preset or fallback.
    --base-url       API URL, or Ollama host (a trailing /v1 is removed).
    --api-key        Explicit key; named providers also support environment keys.
    --strategy       Registered prompting strategy; defaults to direct.
    --baseline       Run code generation and execution instead of a prompt strategy.
    --reasoning-effort  OpenAI-compatible native effort string.
    --ollama-think    Ollama true/false or a model-specific level string.
    --hf-thinking / --no-hf-thinking  HF chat-template Boolean override.
    --trace          JSONL trace destination.
    --max-tokens     Starting output-token budget; truncation retries may grow it.

    Native reasoning options are omitted by default and apply independently of
    the prompting strategy. Options for a different backend are rejected.
    """
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
            "Starting output-token budget; truncation retries may increase it. "
            "When omitted, use the run or strategy defaults."
        ),
    )
    ap.add_argument(
        "--baseline", action="store_true",
        # Keep help text compatible with consoles using limited encodings.
        help=(
            "Run the code-generation baseline instead of a spec-only strategy: "
            "generate an implementation, execute it, return the real output. "
            "Overrides --strategy. Generated-program failures count as wrong; "
            "provider and harness failures are excluded from accuracy scoring."
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
    """Resolve presets and explicit options into a benchmark configuration.

    Prompting defaults to direct. Native reasoning overrides are passed only
    to their matching backend, without inferring the model's reasoning mode.
    """
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")

    backend = args.backend
    model = args.model
    base_url = args.base_url
    api_key = args.api_key

    # Use the provider's environment key when no explicit key was supplied.
    if not api_key and args.provider:
        api_key = os.environ.get(_PROVIDER_ENV_KEYS.get(args.provider.lower(), ""))

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
        if backend != "ollama":
            raise ValueError(f"--model is required for backend '{backend}'")
        model = "qwen3:4b-instruct"
        print(f"[bench_args] No model specified — using default '{model}'")

    # --- 3. Select the prompting strategy independently of native reasoning ---
    strategy = args.strategy
    if getattr(args, "baseline", False):
        # The baseline generates and executes code; it is a separate condition.
        if strategy is not None:
            print(f"[bench_args] --baseline overrides --strategy {strategy}")
        strategy = BASELINE_NAME
    elif strategy is None:
        strategy = "direct"

    # Reject connection settings unsupported by the selected adapter.
    if backend == "hf" and (base_url or api_key):
        raise ValueError("Backend 'hf' does not accept an API URL or API key")
    if backend == "ollama" and api_key:
        raise ValueError("This Ollama adapter does not accept an API key")

    # --- 4. Build backend kwargs ---
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
        baseline=getattr(args, "baseline", False),
    )


# Conventional environment variable per provider. Ollama needs none.
_PROVIDER_ENV_KEYS = {
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
}

# Strategy name reported for the code-generation baseline.
BASELINE_NAME = "code_gen_baseline"


def build_runtime(cfg: BenchConfig):
    """Construct a benchmark runtime from resolved configuration.

    The code-generation baseline is imported only when selected and supplied
    as an instance, keeping it outside the prediction-strategy registry.
    Starting token budgets are supplied separately on execution requests.
    """
    strategy: Any = cfg.strategy
    if cfg.baseline:
        try:
            from baselines.code_gen_baseline import CodeGenBaseline
        except ModuleNotFoundError as exc:
            if exc.name not in {"baselines", "baselines.code_gen_baseline"}:
                raise
            from benchmarks.baselines.code_gen_baseline import CodeGenBaseline
        strategy = CodeGenBaseline()

    return VirtualRuntime(
        backend=cfg.backend,
        model=cfg.model,
        strategy=strategy,
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
        )


def print_provider_table() -> None:
    """Print configured provider presets and their example model IDs."""
    print("Provider presets (--provider):")
    print()
    for name, preset in PROVIDER_PRESETS.items():
        print(f"  {name:<12} {preset['notes']}")
        for model in preset["models"]:
            print(f"               - {model}")
    print()
