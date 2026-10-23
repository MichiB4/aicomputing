#!/usr/bin/env python3
"""
Run every cell of an experiment manifest.

A cell is one (model x benchmark x strategy). The manifest records item and
invocation selections for comparisons within each benchmark. Results join on
benchmark_item_id. Reusing a selection does not guarantee equal model outputs.

    pilot-v3       python scripts/run_experiment.py --manifest manifests/pilot-v3.json \\
                       --provider groq --model qwen/qwen3.8-27b
    main run       python scripts/run_experiment.py --manifest manifests/main-v2.json \\
                       --provider groq --model <id>

Cells run in separate processes, one per cell, for the same reason the earlier
runners did: a crash, a rate limit, or an OOM in one cell must not take the
rest of the run with it. A failed cell is reported and the run continues.

The code-generation baseline is scheduled only on the benchmarks the manifest
lists for it (HumanEval+ and MBPP+); it is never run on GSM8K or BBH.

This script does NOT decide sample sizes, seeds, or item selection. Those live
in the manifest, built by scripts/make_manifest.py. Each cell resolves the
recorded IDs and verifies selected content before model execution. Incomplete
coverage, missing fingerprints or changed content reject the cell. The direct
benchmark runner has an explicit debug drift override; this orchestrator does
not expose it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PKG = os.path.join(_ROOT, "aicomputing")
_BENCH = os.path.join(_PKG, "benchmarks")
for path in (_BENCH, os.path.join(_PKG, "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from experiment import (  # noqa: E402
    BASELINE_NAME, load_manifest, resolve_model_connection, resolve_reasoning_settings,
    resolve_generation_settings,
)

RUN_BENCHMARK = os.path.join(_BENCH, "run_benchmark.py")

def cells(manifest, models, strategies=None, benchmarks=None, with_baseline=True):
    """Enumerate (model, benchmark, strategy, is_baseline) in a stable order."""
    strategies = strategies or manifest["strategies"]
    benchmarks = benchmarks or list(manifest["benchmarks"])
    baseline = manifest.get("baseline")
    out = []
    for model in models:
        for strategy in strategies:
            for bench in benchmarks:
                out.append((model, bench, strategy, False))
        if with_baseline and baseline:
            for bench in baseline["benchmarks"]:
                if bench in benchmarks:
                    out.append((model, bench, BASELINE_NAME, True))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Run an experiment manifest")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", action="append", default=[],
                    help="Repeatable. Defaults to the models named in the manifest.")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--strategies", nargs="+", default=None)
    ap.add_argument("--benchmarks", nargs="+", default=None)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--out-dir", default=None,
                    help="Default: results/<experiment_id>/")
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug only: first N manifest items per cell.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List the cells and exit without calling any model.")
    ap.add_argument("--allow-unfrozen-model-config", action="store_true",
                    help="Run a model that has no complete model_configs entry "
                         "in the manifest. DEBUG ONLY: connection settings come from "
                         "the CLI; generation uses manifest/strategy defaults, so "
                         "the run is not manifest-reproducible and must not be "
                         "reported.")
    args = ap.parse_args()
    if args.limit is not None and args.limit < 0:
        ap.error("--limit must be non-negative")

    manifest = load_manifest(args.manifest)
    experiment_id = manifest.get("experiment_id", "experiment")
    models = args.model or manifest.get("models") or []
    if not models:
        ap.error("no models: pass --model, or record models in the manifest")

    # Validate all selected models before scheduling any cell.
    model_configs = manifest.get("model_configs", {})
    if not isinstance(model_configs, dict):
        ap.error("model_configs must be an object keyed by model ID")
    unfrozen = [model for model in models if model not in model_configs]
    if unfrozen and not args.allow_unfrozen_model_config:
        ap.error("no frozen model configuration for: " + ", ".join(unfrozen))
    try:
        generation = resolve_generation_settings(manifest.get("generation", {}))
    except (TypeError, ValueError) as exc:
        ap.error(f"generation: {exc}")
    for model in models:
        if model in model_configs:
            config = model_configs[model]
            try:
                connection = resolve_model_connection(config)
                resolve_reasoning_settings(connection["backend"], config.get("reasoning", {}))
                resolve_generation_settings(generation, config)
            except (TypeError, ValueError) as exc:
                ap.error(f"model_configs[{model!r}]: {exc}")
            for field, value in connection.items():
                supplied = getattr(args, field)
                if supplied is not None and supplied != value:
                    ap.error(f"CLI {field} conflicts with the manifest for {model!r}")
            if args.api_key and connection["backend"] != "openai":
                ap.error("--api-key is supported only for the OpenAI-compatible adapter")
    if unfrozen:
        print("WARNING: unfrozen models: " + ", ".join(unfrozen))
        print("This debug run is not manifest-reproducible and must not be reported.")

    out_dir = os.path.abspath(
        args.out_dir or os.path.join(_ROOT, "aicomputing", "results", experiment_id))

    plan = cells(manifest, models, args.strategies, args.benchmarks,
                 with_baseline=not args.no_baseline)
    output_paths, seen_outputs = [], {}
    for cell in plan:
        model, bench, strategy, _ = cell
        safe_model = "".join(c if c.isalnum() or c in "-._" else "-" for c in model)
        out_csv = os.path.join(out_dir, f"{safe_model}__{bench}__{strategy}.csv")
        path_key = os.path.normcase(os.path.abspath(out_csv))
        if path_key in seen_outputs:
            ap.error(f"result path collision: {seen_outputs[path_key]!r} "
                     f"and {cell!r} -> {out_csv}")
        seen_outputs[path_key] = cell
        output_paths.append(out_csv)
    os.makedirs(out_dir, exist_ok=True)

    n_items = sum(e["selected"] for e in manifest["benchmarks"].values())
    print(f"Experiment '{experiment_id}'  seed={manifest['seed']}  "
          f"validation={manifest['validation_mode']}")
    print(f"{len(plan)} cells over {len(models)} model(s); "
          f"{n_items} items per full strategy pass")
    print(f"results -> {out_dir}\n")

    if args.dry_run:
        for i, (model, bench, strat, is_base) in enumerate(plan, 1):
            tag = " [baseline]" if is_base else ""
            print(f"  [{i:>3}/{len(plan)}] {model}  {bench} x {strat}{tag}")
        return 0

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    failed, started = [], time.time()

    for i, (cell, out_csv) in enumerate(zip(plan, output_paths), 1):
        model, bench, strategy, is_baseline = cell
        tag = " [baseline]" if is_baseline else ""
        print(f"[{i}/{len(plan)}] {model}  {bench} x {strategy}{tag} " + "-" * 20, flush=True)

        cmd = [sys.executable, RUN_BENCHMARK,
               "--benchmark", bench, "--manifest", os.path.abspath(args.manifest),
               "--model", model, "--out", out_csv]
        if is_baseline:
            cmd.append("--baseline")
        else:
            cmd += ["--strategy", strategy]
        # Frozen connection/native settings are read directly by the child.
        if model in unfrozen:
            cmd.append("--allow-unfrozen-model-config")
            for flag, value in (("--provider", args.provider), ("--backend", args.backend),
                                ("--base-url", args.base_url)):
                if value is not None:
                    cmd += [flag, value]
        if args.api_key:
            cmd += ["--api-key", args.api_key]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, cwd=_BENCH)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            failed.append((model, bench, strategy))
            print(f"    -> FAILED ({elapsed:.1f}s)\n", flush=True)
        else:
            print(f"    -> done ({elapsed:.1f}s)\n", flush=True)

    print("=" * 60)
    print(f"{len(plan) - len(failed)}/{len(plan)} cells completed "
          f"in {(time.time() - started) / 60:.1f} min")
    if failed:
        print("failed cells:")
        for model, bench, strategy in failed:
            print(f"   {model}  {bench} x {strategy}")
        print("\nRerun just these with --model/--benchmarks/--strategies; the "
              "manifest guarantees they evaluate the same items.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
