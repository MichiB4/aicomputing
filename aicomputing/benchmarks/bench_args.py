"""
Shared argument parsing for AI Computing benchmark scripts.

Adds a --provider flag that auto-configures base_url for known providers,
and auto-derives a sensible strategy default from the model name using the
model registry. Thinking is deliberately NOT auto-derived: each strategy
declares the condition it needs, and --think/--no-think is the only way to
override it.

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

# On networks where something intercepts HTTPS (corporate proxy, antivirus
# TLS inspection), Python's bundled CA store often doesn't trust the
# intercepting cert even though the OS does. truststore makes Python's ssl
# module use the OS certificate store instead — same fix in spirit as
# git's http.sslBackend=schannel. Must run before any HTTPS call (Groq,
# Cerebras, HuggingFace datasets, ...), so it's injected here, first.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass  # optional — only needed on networks with TLS interception

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
from aicomputing.strategies import list_strategies


@dataclass
class BenchConfig:
    backend: str
    model: str
    strategy: str
    think_override: Optional[bool]
    backend_kwargs: Dict[str, Any] = field(default_factory=dict)
    trace: Optional[str] = None
    max_tokens: Optional[int] = None
    # True when --baseline was passed: run the code-generation baseline
    # instead of a spec-only strategy. Kept as a separate flag rather than a
    # registry entry because the baseline generates and executes code, which
    # every strategy is forbidden to do — see build_runtime().
    baseline: bool = False


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
    --strategy  direct | cot | hybrid | confidence | consistency.
                Auto-detected as direct/cot from model if omitted (the
                other three are never auto-selected — pass them explicitly).
    --think / --no-think  Override thinking mode. Auto-detected if omitted.
    --trace     Path to write a JSONL trace file.
    --max-tokens  Override the starting token budget (default: auto per
                  strategy/model). Useful for models with a small
                  tokens-per-minute cap that's below what cot/confidence/
                  consistency's default 8192-token budget needs — e.g.
                  Groq's llama-3.1-8b-instant free tier caps at 6000 TPM,
                  so those strategies 413 there unless capped lower.
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
        choices=list_strategies(),
        help=(
            "Prompting strategy. "
            "Default: 'cot' for thinking models, 'direct' for non-thinking models."
        ),
    )
    ap.add_argument("--trace", default=None,
                    help="Path to write a JSONL trace file.")
    ap.add_argument("--max-tokens", type=int, default=None, dest="max_tokens",
                    help="Override the starting token budget (default: auto per strategy/model).")
    ap.add_argument(
        "--baseline", action="store_true",
        # ASCII only: --help is printed with the console's default codepage,
        # which on Windows is often cp932 and cannot encode an em-dash.
        help=(
            "Run the code-generation baseline instead of a spec-only strategy: "
            "generate an implementation, execute it, return the real output. "
            "NOT AI computing; this is the comparison condition. Overrides "
            "--strategy. A generation that fails to execute scores as wrong "
            "(see baselines/code_gen_baseline.py)."
        ),
    )
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

    # Credentials come from the environment when --api-key is not given.
    #
    # This is the other half of "credentials are environment secrets, never
    # manifest settings": the manifest freezes the provider, and the key for
    # that provider is looked up by its conventional variable name. Without
    # this the key had to be threaded through every subprocess invocation as
    # an argument, and forgetting it produced a full cell of provider_error
    # rows -- correctly classified as infrastructure, but a wasted run.
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
        model = "qwen3:4b-instruct"
        print(f"[bench_args] No model specified — using default '{model}'")

    # --- 3. Auto-detect think default from model ---
    thinking_mode = get_thinking_mode(model)

    if args.think:
        think_override = True
    elif args.no_think:
        think_override = False
    else:
        # None: the STRATEGY decides, not the model family.
        #
        # This previously derived think=True for any THINKING_TOGGLE or
        # ALWAYS_THINKS model, and that value reached ExecuteRequest.think,
        # which strategies treat as an explicit caller override. The effect was
        # that `direct` -- the minimal execution condition, which requests
        # think=False -- silently ran WITH reasoning on exactly the models the
        # experiment targets. Direct vs CoT then compared two thinking
        # conditions, and the arm that defines the baseline was not the
        # condition it claimed to be.
        #
        # A model family's capability must not choose the experimental
        # condition. resolve_thinking() decides only whether the backend can
        # HONOUR what the strategy asked for, and records the answer.
        think_override = None

    # --- 4. Auto-detect strategy from model and think setting ---
    strategy = args.strategy
    if getattr(args, "baseline", False):
        # --baseline replaces the strategy entirely. Reported under the
        # baseline's own name so the results CSV and the table generators
        # can tell the comparison condition apart from the spec-only arms.
        if strategy is not None:
            print(f"[bench_args] --baseline overrides --strategy {strategy}")
        strategy = BASELINE_NAME
    elif strategy is None:
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
        max_tokens=args.max_tokens,
        baseline=getattr(args, "baseline", False),
    )


# Name reported in the `strategy` column when --baseline is used. It is the
# CodeGenBaseline's own .name, kept in one place so the table generators and
# the oracles agree on the spelling.
# Conventional environment variable per provider. Ollama needs none.
_PROVIDER_ENV_KEYS = {
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
}

BASELINE_NAME = "code_gen_baseline"


def build_runtime(cfg: BenchConfig):
    """
    Construct the VirtualRuntime for a benchmark run.

    Shared by all four oracles so that --baseline behaves identically in each.

    The baseline is passed as a Strategy *instance* rather than by name.
    VirtualRuntime accepts either, and this keeps the code-generation path out
    of the strategy registry entirely: `create_strategy("code_gen_baseline")`
    still fails, `list_strategies()` still contains only spec-only arms, and
    the one component that executes model-written code stays reachable solely
    by an explicit opt-in flag. That matches the standing rule that strategies
    predict results and never generate or execute code.
    """
    from aicomputing import VirtualRuntime

    strategy: Any = cfg.strategy
    if cfg.baseline:
        try:  # running as `python benchmarks/x_oracle.py` (benchmarks/ on path)
            from baselines.code_gen_baseline import CodeGenBaseline
        except ImportError:  # imported as a package, e.g. from tests
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
    """
    Run one example, converting a failed call into a failed *row*.

    Without this, any exception from vr.execute() propagates out of the
    per-example loop and aborts the whole benchmark cell, discarding every
    example already completed in it. That is the wrong trade for a metric
    that is an average over examples: a single transient provider error
    costs the entire cell rather than one row of it.

    This was not hypothetical. On the Groq pilot, `bbh x cot` died 312
    seconds in on a 429 whose own message said "try again in 255ms",
    throwing away 20 graded examples across four sub-tasks.

    The failure is returned as an ExecuteResult with result=None so that
    callers need no special case: the value compares unequal to any
    expected answer and scores incorrect, which is the honest reading of
    "the model was asked and no answer came back". raw_text carries the
    exception for the CSV, and exec_status is set to "error" so failed
    rows can be counted and excluded rather than silently depressing an
    accuracy figure.

    Note the asymmetry with the retries inside the backend: those retry the
    *same* call because it may yet succeed. This catches what is left after
    they are exhausted, and does not retry — a cell that is hitting the rate
    limit hard should finish and be visibly degraded, not spin.
    """
    import time

    from aicomputing.request import ExecuteResult

    started = time.perf_counter()
    try:
        return vr.execute(req)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # noqa: BLE001 - deliberately total
        detail = f"{type(exc).__name__}: {exc}"
        print(f"    [example failed] {label or ''} {detail}".rstrip(), flush=True)
        # The numeric fields must be real numbers, not None. ExecuteResult
        # defaults latency_ms and tokens_used to None, and every oracle's CSV
        # writer formats latency with `:.0f` -- so a None here crashes the
        # writer at the END of the cell and loses the whole file, which is the
        # exact failure this function exists to prevent.
        #
        # latency is the true elapsed time rather than 0: the retries inside
        # the backend can spend a minute or more before giving up, and that
        # cost is real and worth seeing. tokens_used is 0 because no completion
        # came back. exec_status="error" is what marks the row as a failure;
        # the numbers are not a claim that anything succeeded.
        # Classified as INFRASTRUCTURE, not as a model failure. Two things
        # depend on getting this right:
        #
        #   * exec_status is the code-generation baseline's channel, and
        #     "error" there now means "the generated program raised". Reusing
        #     it for a 429 would file an API outage as bad generated code.
        #   * scorable=False keeps the invocation out of the capability
        #     denominator entirely. A rate limit is not the model answering
        #     incorrectly, and counting it as one turns a quota problem into a
        #     capability claim -- on exactly the expensive arms most likely to
        #     be throttled.
        from aicomputing.request import FailureType
        return ExecuteResult(
            result=None,
            raw_text=f"[execute failed] {detail}",
            model=getattr(vr, "model", ""),
            strategy=getattr(getattr(vr, "strategy", None), "name", ""),
            failure_type=(FailureType.TIMEOUT if isinstance(exc, TimeoutError)
                          else FailureType.PROVIDER_ERROR),
            scorable=False,
            validation_status="unparsed",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            tokens_used=0,
            token_retries=0,
        )


# describe_prompt() was removed here.
#
# It reconstructed the logged prompt from the strategy NAME, which could not be
# made faithful:
#   * it fell through to the direct prompt for any strategy it did not know, so
#     `pseudocode` logged a prompt it had never sent;
#   * it returned one string, so `contract` logged only its first of two stages;
#   * it had no way to see that a validation retry had wrapped the prompt via
#     build_retry_prompt, so every arm was wrong on a retried example.
#
# Each failure was silent and produced plausible-looking text, which is the
# worst property for a column feeding prompt-level analysis. Strategies now
# record what they actually sent in StrategyResult.prompts, which reaches the
# CSV as the `prompts` column and the JSONL trace as "prompts". A strategy
# added later gets correct logging without touching this module.


def print_provider_table() -> None:
    """Print a table of available providers and their models."""
    print("Available providers (--provider):")
    print()
    for name, preset in PROVIDER_PRESETS.items():
        print(f"  {name:<12} {preset['notes']}")
        for m in preset["models"]:
            print(f"               • {m}")
    print()
