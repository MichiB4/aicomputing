"""Configuration for the evaluation command; no inference occurs here."""
from __future__ import annotations

import argparse
from dataclasses import replace
from decimal import Decimal
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from evaluation.benchmarks import BENCHMARKS
from evaluation.benchmarks.result_schema import FAILURE_TYPES
from evaluation.model_profiles import get_model_profile

STRATEGIES = ("direct", "cot", "pseudocode", "contract", "code_execute")
DEFAULTS = dict(model="gpt-6-luna", provider=None, benchmarks=list(BENCHMARKS), strategies=list(STRATEGIES),
                samples=100, seed=42, reasoning="none", temperature=None, top_p=None,
                max_output_tokens=None, timeout=None, retries=1, execution_timeout=None,
                execution_retries=None, budget=20.0, dataset_release=None, base_url=None,
                input_price=None, output_price=None, cached_input_price=None, cache_write_price=None,
                provider_output_tokens=None)
# Custom models inherit connection and capacity defaults from these profiles.
PROVIDER_BASE_MODELS = {"openai": "gpt-6-luna", "cerebras": "qwen-3.8-27b", "ollama": "qwen3:0.6b"}
# Resuming may change operational allowances, never the experimental condition.
MUTABLE = {"timeout", "retries", "execution_timeout", "execution_retries", "budget"}


def parser():
    p = argparse.ArgumentParser(description="Run selected benchmarks and approaches; save only OUTPUT.csv and OUTPUT.json.")
    p.add_argument("--config", help="Optional JSON settings; explicit command-line flags take precedence")
    p.add_argument("--output", help="Output basename, for example results/luna-max")
    p.add_argument("--model")
    p.add_argument("--provider", choices=tuple(PROVIDER_BASE_MODELS))
    p.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS)
    p.add_argument("--strategies", "--approaches", nargs="+", choices=STRATEGIES)
    p.add_argument("--samples", type=int, help="Tasks PER benchmark; any positive size within the approved pool")
    p.add_argument("--seed", type=int)
    p.add_argument("--reasoning", choices=("none", "low", "medium", "high", "xhigh", "max"))
    p.add_argument("--temperature", type=float)
    p.add_argument("--top-p", type=float)
    p.add_argument("--max-output-tokens", type=int, help="Optional per-call cap, including native reasoning tokens")
    p.add_argument("--timeout", type=float, help="HTTP phase timeout in seconds; defaults to 900 with reasoning, otherwise 180")
    p.add_argument("--retries", type=int, help="Additional attempts on temporary API failures (default: 1)")
    p.add_argument("--execution-timeout", type=float, help="Code worker timeout; defaults to --timeout")
    p.add_argument("--execution-retries", type=int, help="Code worker timeout retries; defaults to --retries")
    p.add_argument("--budget", type=float, help="Total conservative USD spending guard, including all prior attempts (default: 20)")
    p.add_argument("--dataset-release", help="Reviewed dataset; defaults to the bundled data/dataset-release.json")
    p.add_argument("--base-url")
    p.add_argument("--key-file", help="Credential file; otherwise use the provider's environment variable")
    for name in ("input-price", "output-price", "cached-input-price", "cache-write-price"):
        p.add_argument("--" + name, type=float, help="USD per million tokens; overrides the model's recorded price")
    p.add_argument("--provider-output-tokens", type=int, help="Known provider output maximum; required for an uncapped custom model")
    action = p.add_mutually_exclusive_group()
    action.add_argument("--resume", action="store_true", help="Continue unfinished evaluations in these same two files")
    action.add_argument("--retry-failed", nargs="*", metavar="FAILURE", choices=("all", *sorted(FAILURE_TYPES)),
                        help="Retry selected failure types; no values means timeout/provider_error; use all for every incorrect result")
    p.add_argument("--dry-run", action="store_true", help="Show settings and evaluation counts without inference or output files")
    return p


def _resolve_path(value, name, directory):
    if not isinstance(value, (str, Path)) or not str(value).strip() or "\x00" in str(value):
        raise ValueError(name + " must be a nonempty path")
    return str((directory / value).resolve())


def read_options(argv=None):
    p = parser()
    args = p.parse_args(argv)
    supplied = {}
    config_dir = Path.cwd()
    if args.config is not None:
        args.config = Path(_resolve_path(args.config, "config", config_dir))
        config_dir = args.config.parent
        supplied = json.loads(args.config.read_text(encoding="utf-8-sig"))
        if not isinstance(supplied, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(supplied) - set(DEFAULTS) - {"output", "key_file"}
        if unknown:
            raise ValueError("Unknown configuration keys: " + ", ".join(sorted(unknown)))
    for key in (*DEFAULTS, "output", "key_file"):
        value = getattr(args, key, None)
        if value is not None:
            supplied[key] = value
    for name in ("output", "key_file", "dataset_release"):
        if supplied.get(name) is not None:
            directory = Path.cwd() if getattr(args, name, None) is not None else config_dir
            supplied[name] = _resolve_path(supplied[name], name, directory)
    output = supplied.pop("output", None)
    key_file = supplied.pop("key_file", None)
    if not output:
        p.error("Provide --output or output in the configuration file")
    out = Path(output).resolve()
    if out.suffix.lower() in (".csv", ".json"):
        out = out.with_suffix("")
    args.output = out
    args.key_file = Path(key_file) if key_file else None
    return args, supplied


def configure(supplied, previous=None):
    settings = dict(previous or DEFAULTS)
    for name, value in supplied.items():
        settings[name] = value
    if not isinstance(settings["model"], str) or not settings["model"].strip():
        raise ValueError("Model must be a nonempty string")
    if settings["dataset_release"] is not None:
        settings["dataset_release"] = _resolve_path(settings["dataset_release"], "dataset_release", Path.cwd())
    if settings["timeout"] is None:
        settings["timeout"] = 180.0 if settings["reasoning"] == "none" else 900.0
    if settings["execution_timeout"] is None:
        settings["execution_timeout"] = settings["timeout"]
    if settings["execution_retries"] is None:
        settings["execution_retries"] = settings["retries"]
    if previous:
        # The shared timeout/retry flags also update worker settings unless specified separately.
        if "timeout" in supplied and "execution_timeout" not in supplied:
            settings["execution_timeout"] = settings["timeout"]
        if "retries" in supplied and "execution_retries" not in supplied:
            settings["execution_retries"] = settings["retries"]
    for name in ("samples", "seed", "retries", "execution_retries", "max_output_tokens", "provider_output_tokens"):
        value = settings[name]
        if value is None and name in ("max_output_tokens", "provider_output_tokens"):
            continue
        if type(value) is not int:
            raise ValueError(name + " must be an integer")
    if settings["samples"] < 1 or any(settings[k] < 0 for k in ("retries", "execution_retries")):
        raise ValueError("Sample size must be positive and retries nonnegative")
    for name in ("max_output_tokens", "provider_output_tokens"):
        if settings[name] is not None and settings[name] < 1:
            raise ValueError(name + " must be positive")
    for name in ("timeout", "execution_timeout", "budget"):
        value = settings[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be finite and positive")
    for name in ("benchmarks", "strategies"):
        allowed = BENCHMARKS if name == "benchmarks" else STRATEGIES
        values = settings[name]
        if (not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values)
                or len(set(values)) != len(values) or any(v not in allowed for v in values)):
            raise ValueError("Invalid or duplicate " + name)
    if settings["reasoning"] not in ("none", "low", "medium", "high", "xhigh", "max"):
        raise ValueError("Invalid reasoning effort")
    provider = settings["provider"]
    if provider is not None and (not isinstance(provider, str) or provider not in PROVIDER_BASE_MODELS):
        raise ValueError("Invalid provider")
    if settings["base_url"] is not None and (not isinstance(settings["base_url"], str) or not settings["base_url"].strip()):
        raise ValueError("base_url must be a nonempty string")
    try:
        profile = get_model_profile(settings["model"])
    except ValueError:
        if provider not in PROVIDER_BASE_MODELS:
            raise ValueError("Custom models require --provider") from None
        if provider != "ollama" and any(settings[k] is None for k in ("input_price", "output_price")):
            raise ValueError("Custom hosted models require --input-price and --output-price (USD per million)") from None
        if settings["provider_output_tokens"] is None and settings["max_output_tokens"] is None:
            raise ValueError("Custom models require --provider-output-tokens or --max-output-tokens") from None
        base = get_model_profile(PROVIDER_BASE_MODELS[provider])
        profile = replace(base, model=settings["model"])
    if settings["provider"] not in (None, profile.provider):
        raise ValueError("Provider conflicts with the model profile")
    settings["provider"] = profile.provider
    settings["base_url"] = settings["base_url"] or profile.api_base
    try:
        endpoint = urlsplit(settings["base_url"])
        _ = endpoint.port  # Validate an explicit port before constructing the client.
    except ValueError:
        raise ValueError("Invalid API URL") from None
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("API URL must not contain credentials, query parameters or fragments")
    if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
        raise ValueError("Invalid API URL")
    if profile.provider == "ollama" and endpoint.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Local models require a loopback API URL")
    if settings["reasoning"] != "none":
        if profile.provider != "openai":
            raise ValueError("Native reasoning levels are currently supported through OpenAI")
        if settings["temperature"] is not None or settings["top_p"] is not None:
            raise ValueError("Omit temperature and top_p when native reasoning is enabled")
    else:
        if profile.model == "gpt-6.1-sol":
            raise ValueError("gpt-6.1-sol requires --reasoning low or higher")
        settings["temperature"] = 0.0 if settings["temperature"] is None else settings["temperature"]
        settings["top_p"] = 1.0 if settings["top_p"] is None else settings["top_p"]
    for name, upper in (("temperature", 2), ("top_p", 1)):
        value = settings[name]
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= upper):
            raise ValueError("Invalid " + name)
    rates = {}
    for name in ("input", "output", "cached_input", "cache_write"):
        value = settings[name + "_price"]
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("Invalid price: " + name)
        rates[name + "_rate"] = getattr(profile, name + "_rate") if value is None else Decimal(str(value)) / 1_000_000
        if not rates[name + "_rate"].is_finite() or rates[name + "_rate"] < 0:
            raise ValueError("Invalid price: " + name)
    if settings["input_price"] is not None:
        for name in ("cached_input", "cache_write"):
            if settings[name + "_price"] is None:
                rates[name + "_rate"] = rates["input_rate"]
    maximum = settings["provider_output_tokens"] or profile.output_reservation
    if rates["cache_write_rate"] < max(rates["input_rate"], rates["cached_input_rate"]):
        raise ValueError("Cache-write reservation rate must cover input and cached-input rates")
    if settings["max_output_tokens"] is not None:
        if settings["max_output_tokens"] > maximum:
            raise ValueError("Token cap exceeds the recorded provider output maximum")
        maximum = min(maximum, settings["max_output_tokens"])
    profile = replace(profile, api_base=settings["base_url"], reasoning_effort=settings["reasoning"],
                      api_format="responses" if settings["reasoning"] != "none" else "chat_completions",
                      output_reservation=maximum, **rates)
    if previous:
        for name in DEFAULTS:
            if name not in MUTABLE and settings[name] != previous.get(name):
                raise ValueError("Existing run setting cannot change: " + name + "; use another --output")
    return settings, profile


def find_release(root, explicit=None):
    """Return the supplied release, or the workspace release if it covers every benchmark."""
    if explicit:
        return Path(explicit).resolve(strict=True)
    path = root / "data/dataset-release.json"
    if not path.is_file():
        raise ValueError("Bundled dataset is missing; restore data/dataset-release.json or supply --dataset-release")
    from evaluation.benchmarks.prepared_data import load_release
    data, _ = load_release(path)
    if not set(BENCHMARKS) <= set(data["benchmarks"]):
        raise ValueError(f"{path} does not cover every benchmark; supply --dataset-release")
    return path.resolve()
