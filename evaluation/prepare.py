"""Prepare a new, portable Luna experiment bundle without inference."""
from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal
from importlib.metadata import PackageNotFoundError
from pathlib import Path

from evaluation import runner
from evaluation.reproducibility import capture_environment, digest, load_inputs, requirements_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-release", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--cases-per-benchmark", type=int, choices=(5, 100), default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budget-usd", type=Decimal, required=True)
    parser.add_argument("--user-ceiling-usd", type=Decimal, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    output = args.out_dir.resolve()
    if output.exists():
        raise FileExistsError("Refusing to replace an existing prepared bundle")
    if workspace / "manifests/prepared" not in output.parents:
        raise ValueError("Prepared bundles must be new subdirectories of manifests/prepared/")
    if not args.experiment_id.strip():
        raise ValueError("experiment-id must not be empty")
    budget, ceiling = args.budget_usd, args.user_ceiling_usd
    if not budget.is_finite() or not ceiling.is_finite() or not Decimal(0) < budget <= ceiling:
        raise ValueError("Require finite 0 < budget <= user ceiling")
    for relative in ("aicomputing/src", "aicomputing/benchmarks"):
        sys.path.insert(0, str(workspace / relative))
    os.environ.update(HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      TOKENIZERS_PARALLELISM="false")
    from experiment import build_manifest, _describe_pool_policy
    from aicomputing.prompting import FRAMED_PROMPT_REVISION
    release = args.dataset_release.resolve(strict=True)
    release_bytes = release.read_bytes()
    environment = capture_environment()
    requirements = requirements_bytes(environment)
    # Freeze the same 100-case selection first; the smoke test uses its prefix.
    manifest = build_manifest(
        experiment_id=args.experiment_id, models=[runner.MODEL],
        model_configs={runner.MODEL: {
            "backend": "openai", "base_url": runner.API_BASE,
            "reasoning": {"reasoning_effort": "none"},
        }},
        strategies=list(runner.STRATEGIES), benchmarks=runner.BENCHMARKS,
        sizes={name: 100 for name in runner.BENCHMARKS}, seed=args.seed,
        validation_mode="strict", generation={"temperature": 0.0, "top_p": 1.0, "max_retries": 0, "max_new_tokens": None},
        dataset_release=str(release), answer_format=runner.ANSWER_FORMAT,
    )
    if manifest["dataset_release"]["sha256"] != digest(release_bytes):
        raise ValueError("Dataset changed during preparation")
    cases = args.cases_per_benchmark
    for entry in manifest["benchmarks"].values():
        entry["items"] = entry["items"][:cases]
        entry.update(requested=cases, selected=cases,
                     invocation_pool_policy=_describe_pool_policy(entry["items"]))
    manifest["dataset_release"]["path"] = "dataset-release.json"
    manifest["prompt_revision"] = FRAMED_PROMPT_REVISION
    manifest["reproducibility"] = {
        "format_version": 1, "environment": environment,
        "requirements_sha256": digest(requirements),
    }
    manifest["experiment_protocol"] = {
        "runner_protocol": runner.RUNNER_PROTOCOL,
        "source_sha256": runner.source_hashes(workspace),
        "runner_sha256": runner.sha256(workspace / "evaluation/runner.py"),
        "uniform_http_phase_timeout_seconds": runner.HTTP_PHASE_TIMEOUT_SECONDS,
        "timeout_retries": runner.TIMEOUT_RETRIES, "sdk_retries": 0, "validation_retries": 0,
        "application_output_cap": None, "native_reasoning_effort": "none", "tools_supplied": False,
        "answer_format": runner.ANSWER_FORMAT,
        "evaluations": 3 * cases * 4, "nominal_model_calls": 3 * cases * 5,
        "max_generation_attempts": 3 * cases * 5 * (runner.TIMEOUT_RETRIES + 1),
        "selection_policy": "seeded 100-case selection; five-case smoke uses its sorted prefix",
        "new_unseen_holdout": False,
        "pricing": runner.pricing_snapshot(),
        "cost_guard": {"operational_usd": str(budget), "user_ceiling_usd": str(ceiling)},
    }
    output.mkdir(parents=True, exist_ok=False)
    # Write the manifest last: its presence follows the complete input files.
    for name, raw in (
        ("dataset-release.json", release_bytes), ("requirements.txt", requirements),
        ("manifest.json", (json.dumps(manifest, ensure_ascii=True, indent=2) + "\n").encode("utf-8")),
    ):
        with (output / name).open("xb") as stream:
            stream.write(raw)
    load_inputs(output / "manifest.json")
    print(f"Prepared {manifest['experiment_protocol']['evaluations']} evaluations; no model calls.")
    print(f"Manifest: {output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, PackageNotFoundError) as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
