"""Run one model/benchmark/strategy cell from a frozen manifest.

Each core function contributes one seeded invocation, shared across models and
strategies. The manifest controls validation and generation; rows use the shared
schema. Debug overrides break reproducibility.

Usage:
    python aicomputing/benchmarks/run_benchmark.py --benchmark gsm8k \\
        --manifest PATH_TO_MANIFEST.json \\
        --provider groq --model qwen/qwen3.8-27b --strategy direct \\
        --out results/experiment/gsm8k_direct.csv"""

from __future__ import annotations

import argparse
from collections import Counter
import sys

from bench_args import (  # adjusts sys.path first
    add_common_args, resolve_args, build_runtime, execute_resilient,
    configure_system_certificates,
)

# Configure application HTTPS before importing clients; grader imports stay inert.
if __name__ == "__main__":
    configure_system_certificates()

from aicomputing import ExecuteRequest
from aicomputing.request import FailureType
from aicomputing.validation import ValidationMode

from adapters import get_adapter
from experiment import (
    BENCHMARK_GROUP,
    check_pools,
    load_manifest,
    resolve_reasoning_settings,
    resolve_model_connection,
    resolve_generation_settings,
    verify_content,
)
from result_schema import build_row, new_run_id, write_results

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - progress bar is cosmetic
    def tqdm(it, **kw):
        return it


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one manifest cell")
    add_common_args(ap)
    ap.add_argument("--benchmark", required=True, choices=sorted(BENCHMARK_GROUP))
    ap.add_argument("--manifest", required=True, help="Path to the experiment manifest JSON")
    ap.add_argument("--out", dest="out_csv", default=None, help="Where to write the result CSV")
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug only: evaluate just the first N manifest items. "
                         "Breaks paired comparability; never use for a reported run.")
    ap.add_argument("--allow-content-drift", action="store_true",
                    help="Downgrade a manifest/data content mismatch from an "
                         "error to a warning. The run then evaluates data the "
                         "manifest did not select, so results are not "
                         "reproducible; debugging only.")
    ap.add_argument("--override-generation", action="store_true",
                    help="Use CLI generation settings instead of the "
                         "manifest's. Off by default: once a manifest exists "
                         "it is authoritative, so a run cannot silently "
                         "disagree with the configuration it records.")
    ap.add_argument(
        "--allow-unfrozen-model-config", action="store_true",
        help="Debug only: allow a model absent from model_configs; not reproducible.",
    )
    args = ap.parse_args()
    if args.limit is not None and args.limit < 0:
        ap.error("--limit must be non-negative")
    manifest = load_manifest(args.manifest)
    if manifest.get("data_policy") == "reviewed_release_v1" and args.allow_content_drift:
        ap.error("Content drift cannot be waived for a reviewed dataset release")
    model_configs = manifest.get("model_configs", {})
    if not isinstance(model_configs, dict):
        raise ValueError("model_configs must be an object keyed by model ID")
    if args.model is None:
        models = manifest.get("models") or []
        if len(models) != 1:
            ap.error("--model is required unless the manifest names exactly one model")
        args.model = models[0]
    if args.model not in model_configs:
        if not args.allow_unfrozen_model_config:
            ap.error("Model has no frozen configuration; use the debug flag explicitly")
        model_cfg = {}
        print("WARNING: unfrozen model configuration; this is a debug run")
    else:
        model_cfg = model_configs[args.model]
        connection = resolve_model_connection(model_cfg)
        for field, value in connection.items():
            supplied = getattr(args, field)
            if supplied is not None and supplied != value:
                raise ValueError(f"CLI {field} conflicts with the manifest")
            setattr(args, field, value)
    cfg = resolve_args(args)
    cli_reasoning = {
        key: cfg.backend_kwargs[key]
        for key in ("reasoning_effort", "think", "enable_thinking")
        if key in cfg.backend_kwargs
    }
    reasoning = resolve_reasoning_settings(
        cfg.backend, model_cfg.get("reasoning", {}), cli_reasoning,
    )
    cfg.backend_kwargs.update(reasoning)
    print(f"Reasoning overrides from manifest (used for this run): {reasoning}")
    resolved_gen = resolve_generation_settings(manifest.get("generation", {}), model_cfg)
    if cfg.max_tokens is not None:
        if cfg.max_tokens <= 0:
            raise ValueError("--max-tokens must be positive")
        if not args.override_generation and cfg.max_tokens != resolved_gen["max_new_tokens"]:
            raise ValueError(
                "--max-tokens conflicts with the manifest; use --override-generation "
                "for an explicit debug override"
            )
    if args.override_generation:
        print("NOTE: --override-generation set; CLI values take precedence "
              "over the manifest and this run is not manifest-reproducible.",
              file=sys.stderr)
        if cfg.max_tokens is not None:
            resolved_gen["max_new_tokens"] = cfg.max_tokens

    benchmark = args.benchmark

    drift = check_pools(manifest, benchmark)
    if drift:
        print(f"WARNING: {drift}", file=sys.stderr)

    adapter = get_adapter(benchmark)
    items = adapter.resolve(manifest)

    # Positional IDs alone do not identify unchanged data; verify frozen content.
    problems = verify_content(manifest, benchmark, items,
                              hard=not args.allow_content_drift)
    if problems:
        print(f"WARNING: {len(problems)} item(s) no longer match the manifest; "
              f"results are NOT reproducible. First: {problems[0]}", file=sys.stderr)

    if args.limit is not None:
        items = items[: args.limit]

    strategy_name = cfg.strategy
    with build_runtime(cfg) as vr:
        output_mode = ValidationMode.coerce(
            manifest.get("validation_mode"), vr.validation_mode,
        )
        run_id = new_run_id(cfg.model, benchmark, strategy_name)

        print(f"{benchmark} [{BENCHMARK_GROUP[benchmark]}] x {strategy_name} "
              f"on {cfg.model}: {len(items)} items "
              f"(validation={manifest.get('validation_mode')}, seed={manifest.get('seed')})")

        rows = []
        for item in tqdm(items, desc=benchmark):
            result = execute_resilient(
                vr,
                ExecuteRequest(
                    signature=item.signature,
                    docstring=item.docstring,
                    inputs=item.inputs,
                    return_type=item.return_type,
                    strategy=cfg.strategy,
                    temperature=resolved_gen["temperature"],
                    top_p=resolved_gen["top_p"],
                    max_retries=resolved_gen["max_retries"],
                    max_new_tokens=resolved_gen["max_new_tokens"],
                    answer_format=manifest.get("answer_format", "legacy"),
                    # The manifest, not the caller, decides the policy, so every
                    # cell in one experiment is validated identically.
                    validation_mode=manifest.get("validation_mode"),
                ),
                label=f"{benchmark}:{item.item_id}",
            )
            correct = (
                result.validation_status != "unparsed"
                and not (
                    output_mode is ValidationMode.STRICT
                    and result.validation_status in ("invalid_type", "invalid_constraint")
                )
                and adapter.score(result.result, item)
            )
            rows.append(build_row(
                result,
                run_id=run_id,
                benchmark=benchmark,
                item_id=item.item_id,
                signature=item.signature,
                docstring=item.docstring,
                inputs=item.inputs,
                expected=item.expected,
                correct=bool(correct),
                invocation_pool=item.invocation_pool,
                return_type_source=item.return_type_source,
                generation=resolved_gen,
            ))

        # Score capability over scorable outcomes; report infrastructure separately.
        attempted = len(rows)
        scorable_rows = [r for r in rows if r["scorable"]]
        infra_rows = [r for r in rows if not r["scorable"]]
        n_scorable = len(scorable_rows)
        n_correct_scorable = sum(1 for r in scorable_rows if r["correct"])
        acc = n_correct_scorable / n_scorable if n_scorable else 0.0
        infra_rate = len(infra_rows) / attempted if attempted else 0.0

        print("=" * 56)
        print(f"{benchmark} x {strategy_name}")
        print(f"   attempted              : {attempted}")
        print(f"   scorable               : {n_scorable}")
        print(f"   infrastructure failures: {len(infra_rows)}")
        print(f"   correct (of scorable)  : {n_correct_scorable}")
        print(f"   accuracy over scorable : {acc:.4f}")
        print(f"   infra failure rate     : {infra_rate:.4f}")
        if infra_rows:
            print("   NOTE: infrastructure failures are excluded from the "
                  "denominator above; they are not wrong answers.")

        failures = Counter(r["failure_type"] for r in rows if r["failure_type"])
        if failures:
            print("failure breakdown:")
            for kind, count in failures.most_common():
                tag = "" if kind in FailureType.SCORABLE else "   [infrastructure]"
                print(f"   {count:4}  {kind}{tag}")


        if args.out_csv:
            write_results(args.out_csv, rows)
            print(f"Results written to {args.out_csv}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
