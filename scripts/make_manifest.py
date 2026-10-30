#!/usr/bin/env python3
"""Freeze reviewed three-benchmark selections and model settings.

For the audited Luna workflow use python -m evaluation.prepare. This CLI also
supports manifests for future local models through explicit model configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
    build_manifest,
    save_manifest,
    resolve_model_connection,
    resolve_reasoning_settings,
    resolve_generation_settings,
)
from aicomputing.answer_protocol import ANSWER_FORMATS, FRAMED_V1, FRAMED_TEXT_V1  # noqa: E402
from aicomputing.prompting import FRAMED_PROMPT_REVISION  # noqa: E402

DEFAULT_SIZES = {"humaneval_plus": 100, "mbpp_plus": 100, "gsm8k": 100}
DEFAULT_STRATEGIES = ["direct", "cot", "pseudocode", "contract"]



def main() -> int:
    ap = argparse.ArgumentParser(description="Build an experiment manifest")
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--dataset-release", required=True, help="Approved local data release; no network lookup.")
    ap.add_argument("--out", default=None, help="Output path; defaults to manifests/<experiment-id>.json")
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
    ap.add_argument("--benchmarks", nargs="+", default=None,
                    choices=list(ALL_BENCHMARKS))
    ap.add_argument("--strategies", nargs="+", default=None,
                    help="Default: direct, cot, pseudocode, contract")
    ap.add_argument("--validation-mode", choices=("strict", "warn"), default="strict")
    ap.add_argument("--answer-format", choices=ANSWER_FORMATS, default=None,
                    help="Response protocol; defaults to framed_text_v1")
    args = ap.parse_args()
    if not args.experiment_id.strip() or Path(args.experiment_id).name != args.experiment_id:
        ap.error("experiment-id must be a nonempty filename component")
    args.benchmarks = args.benchmarks or list(DEFAULT_SIZES)
    args.strategies = args.strategies or list(DEFAULT_STRATEGIES)
    if args.validation_mode != "strict":
        ap.error("Reviewed evaluations require strict validation")

    # resolve the frozen model configuration, if any
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

    out = args.out or os.path.join(_ROOT, "manifests", args.experiment_id + ".json")

    print(f"Building '{args.experiment_id}' manifest (seed={args.seed}) ...")
    manifest = build_manifest(
        experiment_id=args.experiment_id,
        models=args.models,
        strategies=args.strategies,
        sizes=DEFAULT_SIZES,
        benchmarks=args.benchmarks,
        seed=args.seed,
        validation_mode=args.validation_mode,
        model_configs=model_configs,
        dataset_release=args.dataset_release,
        generation={"max_retries": 0},
        answer_format=args.answer_format or FRAMED_TEXT_V1,
    )
    if manifest["answer_format"] in (FRAMED_V1, FRAMED_TEXT_V1):
        manifest["prompt_revision"] = FRAMED_PROMPT_REVISION
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
    print(f"\nstrategies : {', '.join(strategies)}")
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
          + (f", x {len(manifest['models'])} models" if manifest["models"] else ""))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
