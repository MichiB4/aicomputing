"""
Unified benchmark runner: one manifest cell (model x benchmark x strategy).

Replaces the four bespoke `*_oracle.py` main() functions for the revised
experiment. Those remain in the tree as the graders they always were -- this
module imports their scoring functions -- and as the entry points that
reproduce the historical pilot results, which must stay runnable.

What changed relative to running an oracle directly:

  * Items come from the experiment manifest, never from `--limit N`. Every
    model and strategy is handed an identical, seeded item set, which is the
    precondition for the paired analysis.
  * One invocation per function on the core benchmarks.
  * Structural validation runs in the mode the manifest declares (STRICT for
    the experiment), so an invalid final value is an execution failure rather
    than a silently returned object.
  * Every row uses the shared result schema, so results join across
    benchmarks without special-casing.
  * The code-generation baseline is refused outside the core coding
    benchmarks.

Usage:
    python benchmarks/run_benchmark.py --benchmark gsm8k \\
        --manifest manifests/pilot-v2.json \\
        --provider groq --model qwen/qwen3.8-27b --strategy direct \\
        --out results/experiment/pilot-v2_gsm8k_direct.csv
"""

from __future__ import annotations

import argparse
import sys

from bench_args import (  # adjusts sys.path first
    add_common_args, resolve_args, build_runtime, execute_resilient,
)
from aicomputing import ExecuteRequest
from aicomputing.request import FailureType

from adapters import get_adapter
from experiment import (
    BASELINE_BENCHMARKS,
    BASELINE_NAME,
    BENCHMARK_GROUP,
    check_pools,
    load_manifest,
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
    args = ap.parse_args()
    cfg = resolve_args(args)

    manifest = load_manifest(args.manifest)
    benchmark = args.benchmark

    # The baseline answers "would generating an implementation have done
    # better?", which is only meaningful where the item IS a function to
    # implement. Refused rather than warned: a silently-produced GSM8K
    # baseline column would invite exactly the comparison that does not hold.
    if cfg.baseline and benchmark not in BASELINE_BENCHMARKS:
        ap.error(
            f"the {BASELINE_NAME} baseline is defined only on "
            f"{sorted(BASELINE_BENCHMARKS)}, not on {benchmark!r}. On a word "
            f"problem the model would have to invent a program for a one-off "
            f"question, so a failure there measures the framing, not code "
            f"generation."
        )

    drift = check_pools(manifest, benchmark)
    if drift:
        print(f"WARNING: {drift}", file=sys.stderr)

    adapter = get_adapter(benchmark)
    items = adapter.resolve(manifest)

    # Content verification. GSM8K and BBH item IDs are positional, so an ID
    # alone does not identify an observation; the stored fingerprint does.
    # A mismatch means the dataset moved underneath the manifest and this run
    # would silently evaluate different problems while claiming to reproduce
    # the recorded experiment -- a hard error unless explicitly waived.
    problems = verify_content(manifest, benchmark, items,
                              hard=not args.allow_content_drift)
    if problems:
        print(f"WARNING: {len(problems)} item(s) no longer match the manifest; "
              f"results are NOT reproducible. First: {problems[0]}", file=sys.stderr)

    if args.limit:
        items = items[: args.limit]

    # The manifest is authoritative for everything that defines the
    # experimental condition. Previously it recorded these and the runner
    # then used its own defaults, so a manifest could state temperature=0.0
    # while the run used something else and the row would record neither.
    gen = dict(manifest.get("generation") or {})
    # Per-model overrides from the manifest win over the global block. This is
    # where a model's token cap belongs: recorded with the experiment, not in
    # a runner-side table that can change the run without changing the
    # manifest.
    model_cfg = (manifest.get("model_configs") or {}).get(cfg.model, {})
    gen.update({k: v for k, v in model_cfg.items()
                if k in ("temperature", "top_p", "max_retries", "max_new_tokens")})
    if args.override_generation:
        print("NOTE: --override-generation set; CLI values take precedence "
              "over the manifest and this run is not manifest-reproducible.",
              file=sys.stderr)
        if cfg.max_tokens:
            gen["max_new_tokens"] = cfg.max_tokens
    resolved_gen = {
        "temperature": gen.get("temperature", 0.0),
        "top_p": gen.get("top_p", 1.0),
        "max_retries": gen.get("max_retries", 1),
        "max_new_tokens": gen.get("max_new_tokens", cfg.max_tokens),
    }

    strategy_name = BASELINE_NAME if cfg.baseline else cfg.strategy
    vr = build_runtime(cfg)
    run_id = new_run_id(cfg.model, benchmark, strategy_name)
    seed = manifest.get("seed")
    manifest_version = manifest.get("manifest_version")

    print(f"{benchmark} [{BENCHMARK_GROUP[benchmark]}] x {strategy_name} "
          f"on {cfg.model}: {len(items)} items "
          f"(validation={manifest.get('validation_mode')}, seed={seed})")

    rows = []
    n_correct = 0
    for item in tqdm(items, desc=benchmark):
        result = execute_resilient(
            vr,
            ExecuteRequest(
                signature=item.signature,
                docstring=item.docstring,
                inputs=item.inputs,
                return_type=item.return_type,
                strategy=None if cfg.baseline else cfg.strategy,
                think=cfg.think_override,
                temperature=resolved_gen["temperature"],
                top_p=resolved_gen["top_p"],
                max_retries=resolved_gen["max_retries"],
                max_new_tokens=resolved_gen["max_new_tokens"],
                # The manifest, not the caller, decides the policy, so every
                # cell in one experiment is validated identically.
                validation_mode=manifest.get("validation_mode"),
            ),
            label=f"{benchmark}:{item.item_id}",
        )
        correct = adapter.compare(result.result, item.expected)
        n_correct += bool(correct)

        pre_correct = None
        if result.revised is not None:
            pre_correct = adapter.compare(result.pre_revision_value, item.expected)

        rows.append(build_row(
            result,
            run_id=run_id,
            benchmark=benchmark,
            benchmark_group=BENCHMARK_GROUP[benchmark],
            item_id=item.item_id,
            signature=item.signature,
            docstring=item.docstring,
            inputs=item.inputs,
            expected=item.expected,
            correct=bool(correct),
            seed=seed,
            manifest_version=manifest_version,
            category=item.category,
            invocation_id=item.invocation_id,
            input_line=item.input_line,
            is_baseline=cfg.baseline,
            pre_revision_correct=pre_correct,
            dataset_revision=(manifest["benchmarks"][benchmark]
                              .get("dataset_revision")),
            invocation_pool=item.invocation_pool,
            return_type_source=item.return_type_source,
            generation=resolved_gen,
        ))

    from collections import Counter

    # Capability accuracy is over SCORABLE invocations only.
    #
    # A provider outage, a rate limit or a harness bug is not the model
    # answering incorrectly. Leaving those in the denominator lets an
    # infrastructure problem read as a capability result -- and the arms most
    # likely to be throttled are the ones that cost the most calls, so the
    # error is not evenly distributed across the comparison.
    #
    # Infrastructure failures are never hidden: attempted, scorable and the
    # failure rate are reported alongside.
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
