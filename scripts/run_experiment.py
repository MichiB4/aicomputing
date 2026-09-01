#!/usr/bin/env python3
"""
Run every cell of an experiment manifest.

A cell is one (model x benchmark x strategy). The manifest fixes which items
each cell evaluates, so all cells are directly comparable and the results join
on benchmark_item_id.

    pilot-v2       python scripts/run_experiment.py --manifest manifests/pilot-v2.json \\
                       --provider groq --model qwen/qwen3.8-27b
    main run       python scripts/run_experiment.py --manifest manifests/main-v1.json \\
                       --provider groq --model <id>

Cells run in separate processes, one per cell, for the same reason the earlier
runners did: a crash, a rate limit, or an OOM in one cell must not take the
rest of the run with it. A failed cell is reported and the run continues.

The code-generation baseline is scheduled only on the benchmarks the manifest
lists for it (HumanEval+ and MBPP+); it is never run on GSM8K or BBH.

This script does NOT decide sample sizes, seeds, or item selection. Those live
in the manifest, built by scripts/make_manifest.py. Running the same manifest
twice evaluates the same items.
"""

from __future__ import annotations

import argparse
import json
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

from experiment import BASELINE_NAME, load_manifest  # noqa: E402

RUN_BENCHMARK = os.path.join(_BENCH, "run_benchmark.py")

# Per-request token caps keyed by MODEL, because the constraint belongs to the
# model. Only add an entry where a cap was actually observed: an unnecessary
# one truncates the structured arms and corrupts what the experiment measures.
# DEPRECATED as an experiment input. The manifest's `model_configs` block is
# authoritative; this remains only as a convenience default for ad-hoc runs
# against a manifest that has no entry for the model, and the runner prints a
# warning when it is used so the gap is visible rather than silent.
MODEL_MAX_TOKENS = {
    # Groq free tier counts PROMPT + REQUESTED max_tokens against an 8000 TPM
    # bucket for both of these, so a high cap 429s on every call that follows
    # another within the minute. Verified 2026-08-29.
    "openai/gpt-oss-120b": 4096,
    "qwen/qwen3.8-27b": 4096,
}


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
                         "in the manifest. DEBUG ONLY: provider and token cap "
                         "then come from the CLI and a runner-side table, so "
                         "the run is not manifest-reproducible and must not be "
                         "reported.")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    experiment_id = manifest.get("experiment_id", "experiment")
    models = args.model or manifest.get("models") or []
    if not models:
        ap.error("no models: pass --model, or record models in the manifest")

    # --- model configuration must be frozen for a reported run -------------
    #
    # provider + model ID + any required token cap are part of the
    # experimental condition. Previously a manifest without model_configs ran
    # anyway, taking the provider from the CLI and the cap from a runner-side
    # table -- so two runs of the same manifest could differ in ways the
    # manifest did not record, which is exactly what it exists to prevent.
    model_configs = manifest.get("model_configs") or {}
    unfrozen = [m for m in models
                if not (model_configs.get(m) or {}).get("provider")]
    if unfrozen and not args.allow_unfrozen_model_config:
        ap.error(
            "no frozen model_configs entry for: " + ", ".join(unfrozen)
            + ".\n"
            "A reported run must take provider and token limits from the "
            "manifest. Build a finalised manifest first:\n"
            "    python scripts/make_manifest.py --preset "
            + ("pilot" if experiment_id.startswith("pilot") else "main")
            + " --model-configs <config.json>\n"
            "or pass --allow-unfrozen-model-config for a debugging run that "
            "is explicitly NOT reproducible."
        )
    if unfrozen:
        print("!" * 68)
        print("WARNING: running UNFROZEN models: " + ", ".join(unfrozen))
        print("Provider and token caps come from the CLI and runner defaults,")
        print("not from the manifest. This run is NOT manifest-reproducible")
        print("and must not be reported as pilot-v2 or main-v1 output.")
        print("!" * 68)

    # --- provider comes from the manifest for frozen models ----------------
    #
    # A CLI provider that disagrees with the frozen one would silently change
    # the experimental condition while the manifest kept describing the old
    # one, so a mismatch is refused rather than resolved.
    for model in models:
        frozen_provider = (model_configs.get(model) or {}).get("provider")
        if frozen_provider and args.provider and args.provider != frozen_provider:
            ap.error(
                f"--provider {args.provider!r} contradicts the manifest, which "
                f"freezes {model!r} to provider {frozen_provider!r}. Drop "
                f"--provider (it is unnecessary for a frozen run), or rebuild "
                f"the manifest if the provider genuinely changed."
            )

    out_dir = args.out_dir or os.path.join(_ROOT, "aicomputing", "results", experiment_id)
    os.makedirs(out_dir, exist_ok=True)

    plan = cells(manifest, models, args.strategies, args.benchmarks,
                 with_baseline=not args.no_baseline)

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

    for i, (model, bench, strategy, is_baseline) in enumerate(plan, 1):
        tag = " [baseline]" if is_baseline else ""
        print(f"[{i}/{len(plan)}] {model}  {bench} x {strategy}{tag} " + "-" * 20, flush=True)

        safe_model = "".join(c if c.isalnum() or c in "-._" else "-" for c in model)
        out_csv = os.path.join(out_dir, f"{safe_model}__{bench}__{strategy}.csv")

        cmd = [sys.executable, RUN_BENCHMARK,
               "--benchmark", bench, "--manifest", os.path.abspath(args.manifest),
               "--model", model, "--out", out_csv]
        if is_baseline:
            cmd.append("--baseline")
        else:
            cmd += ["--strategy", strategy]
        # Frozen provider wins; --provider is only a fallback for the
        # explicitly unfrozen debug path (and is validated to agree above).
        provider = (model_configs.get(model) or {}).get("provider") or args.provider
        if provider:
            cmd += ["--provider", provider]
        if args.api_key:
            cmd += ["--api-key", args.api_key]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        # Only used when the manifest does not configure this model. A cap
        # applied from here does not appear in the manifest, so the run is not
        # fully manifest-reproducible and says so.
        # A frozen model's cap comes from the manifest, which run_benchmark
        # reads directly, so nothing is passed here. MODEL_MAX_TOKENS is only
        # consulted on the explicitly unfrozen debug path.
        if model not in model_configs:
            cap = MODEL_MAX_TOKENS.get(model)
            if cap:
                print(f"    NOTE: {model} is unfrozen; applying the runner "
                      f"default max_tokens={cap}. Not manifest-reproducible.",
                      flush=True)
                cmd += ["--max-tokens", str(cap)]

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
