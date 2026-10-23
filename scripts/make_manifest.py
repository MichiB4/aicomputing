#!/usr/bin/env python3
"""
Build an experiment manifest: the frozen list of items every cell evaluates.

Selection happens once, here, and is written to disk. Runs read the recorded
item and invocation IDs rather than sampling again. Each cell resolves those
IDs through the benchmark adapters and verifies the selected content before
model execution, unless the explicit debug drift override is used. Results
join on benchmark_item_id; this does not guarantee identical model outputs.

Two configurations ship:

    --preset pilot   small smoke configuration (pilot-v3). Verifies plumbing.
    --preset main    the configured primary-experiment sample sizes.

Examples
--------
    python scripts/make_manifest.py --preset pilot
    python scripts/make_manifest.py --preset main --models a b c d
    python scripts/make_manifest.py --preset main --seed 42 --out manifests/main.json

Matching seeds, ID pools, invocation ordering and selection rules reproduce
the selection. A rebuilt manifest can still differ, including its timestamp.
Dataset metadata describes available source information; it is not proof of
identical resolved content. ID-pool checks detect changes to the ID collection;
selected-content verification detects missing fingerprints, incomplete item
coverage and changes to recorded specifications, inputs or expected outputs.
Default construction fails if selected content cannot be fully fingerprinted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PKG = os.path.join(_ROOT, "aicomputing")
for path in (os.path.join(_PKG, "benchmarks"), os.path.join(_PKG, "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from bench_args import configure_system_certificates  # noqa: E402

# Configure application HTTPS before importing dataset/client modules.
if __name__ == "__main__":
    configure_system_certificates()

from experiment import (  # noqa: E402
    ALL_BENCHMARKS,
    DEFAULT_SEED,
    MAIN_SIZES,
    PILOT_SIZES,
    build_manifest,
    save_manifest,
    resolve_model_connection,
    resolve_reasoning_settings,
    resolve_generation_settings,
)
from aicomputing.strategies import list_arms  # noqa: E402

PRESETS = {
    "pilot": {
        "experiment_id": "pilot-v3",
        "sizes": PILOT_SIZES,
        "default_out": "manifests/pilot-v3.json",
    },
    "main": {
        "experiment_id": "main-v2",
        "sizes": MAIN_SIZES,
        "default_out": "manifests/main-v2.json",
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build an experiment manifest")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="pilot")
    ap.add_argument("--out", default=None, help="Output path (default: per preset)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--model-configs", default=None, metavar="JSON",
                    help="Path to JSON model entries with provider or backend, "
                         "optional base_url/reasoning/generation settings. Supplying it FREEZES the run "
                         "configuration: `models` is derived from its keys and "
                         "`model_configs` is stored in the manifest. Required "
                         "for a reported run; never put API keys in it.")
    ap.add_argument("--models", nargs="+", default=[],
                    help="Model IDs this experiment will cover. Recorded only; "
                         "no availability check is made here, because provider "
                         "catalogs drift and an unverified assumption baked "
                         "into a manifest is worse than none.")
    ap.add_argument("--benchmarks", nargs="+", default=list(ALL_BENCHMARKS),
                    choices=list(ALL_BENCHMARKS))
    ap.add_argument("--strategies", nargs="+", default=list_arms(),
                    help=f"Default: the evaluation arms ({', '.join(list_arms())})")
    ap.add_argument("--validation-mode", choices=("strict", "warn"), default="strict")
    ap.add_argument("--no-baseline", action="store_true",
                    help="Omit the code-generation baseline from the manifest")
    args = ap.parse_args()

    # --- resolve the frozen model configuration, if any --------------------
    model_configs = {}
    if args.model_configs:
        with open(args.model_configs, encoding="utf-8") as fh:
            model_configs = json.load(fh)
        if not isinstance(model_configs, dict) or not model_configs:
            ap.error(f"{args.model_configs} must be a non-empty JSON object "
                     f"keyed by model ID")
        for model_id, cfg in model_configs.items():
            try:
                connection = resolve_model_connection(cfg)
                resolve_reasoning_settings(connection["backend"], cfg.get("reasoning", {}))
                resolve_generation_settings({}, cfg)
            except (TypeError, ValueError) as exc:
                ap.error(f"model_configs[{model_id!r}]: {exc}")
        if args.models and sorted(args.models) != sorted(model_configs):
            ap.error("--models disagrees with the keys in --model-configs; "
                     "supply one or the other")
        args.models = sorted(model_configs)

    preset = PRESETS[args.preset]
    out = args.out or os.path.join(_ROOT, preset["default_out"])

    print(f"Building '{preset['experiment_id']}' manifest (seed={args.seed}) ...")
    manifest = build_manifest(
        experiment_id=preset["experiment_id"],
        models=args.models,
        strategies=args.strategies,
        sizes=preset["sizes"],
        benchmarks=args.benchmarks,
        seed=args.seed,
        validation_mode=args.validation_mode,
        include_baseline=not args.no_baseline,
        model_configs=model_configs,
    )
    save_manifest(manifest, out)

    print(f"\n{'benchmark':16}{'group':16}{'selected':>10}{'pool':>8}")
    print("-" * 52)
    total = 0
    for name, entry in manifest["benchmarks"].items():
        total += entry["selected"]
        print(f"{name:16}{entry['group']:16}{entry['selected']:>10}{entry['pool_size']:>8}")
        if entry.get("strata"):
            for task, count in sorted(entry["strata"].items()):
                print(f"    {task:<44}{count:>4}")
    print("-" * 52)
    print(f"{'TOTAL':16}{'':16}{total:>10}")

    strategies = manifest["strategies"]
    baseline = manifest.get("baseline")
    print(f"\nstrategies : {', '.join(strategies)}")
    print(f"baseline   : {baseline['name'] + ' on ' + ', '.join(baseline['benchmarks']) if baseline else 'none'}")
    print(f"validation : {manifest['validation_mode']}")
    if manifest["model_configs"]:
        print("models     : FROZEN")
        for model_id, cfg in sorted(manifest["model_configs"].items()):
            settings = ", ".join(f"{k}={v}" for k, v in sorted(cfg.items()))
            print(f"   {model_id}: {settings}")
    else:
        print("models     : TEMPLATE (models: [], model_configs: {}) -- this is "
              "a selection template, not a run manifest. Rebuild with "
              "--model-configs before a reported run.")
    print(f"cells      : {len(strategies)} strategies x {len(manifest['benchmarks'])} benchmarks"
          + (f" + baseline x {len(baseline['benchmarks'])}" if baseline else "")
          + (f", x {len(manifest['models'])} models" if manifest["models"] else ""))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
