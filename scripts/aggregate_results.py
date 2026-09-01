#!/usr/bin/env python3
"""
Aggregate per-example benchmark CSVs into the long format make_tables.py reads.

Why this exists
---------------
The oracle scripts write one row per *example* (400 rows for a 400-problem
run). make_tables.py expects one row per *cell*:

    model,benchmark,strategy,n,correct,accuracy
    qwen3:4b-instruct,humaneval,direct,164,71,0.4329

Nothing bridged the two, so make_tables produced empty tables from a directory
full of real results. This script is that bridge: group by
(model, benchmark, strategy), count, divide.

It also emits the derived columns the §6.3/§6.4/§6.5 generators need, all
computed from data the oracles already record — no extra model calls:

    exec_ok, exec_error, exec_timeout   code-generation baseline only (§5.5/§6.3)
    revised, flip_to_correct,           contract only (§6.4)
      flip_to_incorrect
    type_warned                         any arm (§6.5)

Usage
-----
    python scripts/aggregate_results.py --raw results/eval_raw \\
                                        --out results/aggregated/summary.csv
    python scripts/make_tables.py --results results/aggregated --out paper/tables

Scoping
-------
--raw takes ONE directory, deliberately. Pointing a glob at results/ swept up
every stale exploratory CSV and silently averaged them into the reported
numbers. Aggregating a specific run's directory into a separate output
directory keeps what reaches the paper decidable by looking at a path.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# The module docstring below is reused as argparse's --help text and
# contains non-ASCII (em dashes, section marks). Windows consoles often
# default to a codepage that cannot encode those, which makes --help
# crash. Force UTF-8, as the runner scripts already do.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Per-example columns the aggregator relies on. `benchmark` is recent; older
# CSVs are recovered from the filename instead (see infer_benchmark).
def _is_scorable(row) -> bool:
    """
    Whether a row belongs in the capability denominator.

    Prefers the explicit `scorable` column. Rows predating it are treated as
    scorable, which is what they were: the column was added with the failure
    taxonomy, and every earlier row recorded a model outcome.
    """
    raw = row.get("scorable")
    if raw is None or str(raw).strip() == "":
        return True
    return str(raw).strip().lower() in ("true", "1", "yes")


REQUIRED = ["correct", "model", "strategy"]

# The revised experiment. hotpotqa is gone; the core benchmarks are the
# EvalPlus pair. Legacy names are kept so historical pilot CSVs still
# aggregate rather than being silently dropped.
KNOWN_BENCHMARKS = ["humaneval_plus", "mbpp_plus", "gsm8k", "bbh"]
LEGACY_BENCHMARKS = ["humaneval", "hotpotqa"]

# Filenames written by the runner scripts:
#   eval_{benchmark}_{model-slug}_{strategy}.csv
#   quick_{benchmark}_{strategy}.csv
_FILENAME_RE = re.compile(
    # Legacy names are included deliberately: historical pilot CSVs must still
    # aggregate. Dropping them from the pattern silently reclassified those
    # files as "unknown benchmark" instead of failing loudly.
    # Longest-first so `humaneval_plus` is not shadowed by `humaneval`.
    r"^(?:eval|quick)_("
    + "|".join(sorted(KNOWN_BENCHMARKS + LEGACY_BENCHMARKS, key=len, reverse=True))
    + r")_", re.I
)


def infer_benchmark(path: Path) -> str | None:
    """Recover the benchmark from a filename for CSVs written before the
    `benchmark` column existed. Returns None rather than guessing."""
    m = _FILENAME_RE.match(path.name)
    return m.group(1).lower() if m else None


def truthy(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def is_blank(value) -> bool:
    return value is None or str(value).strip() == ""


def load_examples(raw_dir: Path) -> tuple[list[dict], list[str]]:
    """Read every per-example CSV in one directory. Returns (rows, problems)."""
    rows: list[dict] = []
    problems: list[str] = []
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        problems.append(f"no CSV files in {raw_dir}")
        return rows, problems

    for path in files:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
            if missing:
                problems.append(
                    f"{path.name}: missing column(s) {', '.join(missing)} — skipped. "
                    f"Is this a per-example results CSV?"
                )
                continue
            fallback = infer_benchmark(path)
            n_before = len(rows)
            for row in reader:
                bench = row.get("benchmark") or fallback
                if not bench:
                    continue
                row["benchmark"] = bench.lower()
                row["_source"] = path.name
                rows.append(row)
            if len(rows) == n_before:
                problems.append(f"{path.name}: no usable rows")
            elif not (reader.fieldnames and "benchmark" in reader.fieldnames):
                problems.append(
                    f"{path.name}: no `benchmark` column; inferred '{fallback}' "
                    f"from the filename"
                )
    return rows, problems


def aggregate(rows: list[dict]) -> list[dict]:
    """Group per-example rows into one row per (model, benchmark, strategy)."""
    cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["benchmark"], r["strategy"])].append(r)

    out: list[dict] = []
    for (model, bench, strat), group in sorted(cells.items()):
        attempted = len(group)

        # --- capability accuracy is over SCORABLE invocations only ----------
        #
        # A provider outage, a rate limit or a harness bug is not the model
        # answering incorrectly, and leaving those in the denominator lets an
        # infrastructure problem read as a capability result -- on the arms
        # that cost the most calls, which are the ones most likely to be
        # throttled. They are reported separately, never hidden.
        #
        # Rows written before `scorable` existed have no such column; those
        # are treated as scorable, which matches how they were produced.
        scorable_rows = [r for r in group if _is_scorable(r)]
        infra_rows = [r for r in group if not _is_scorable(r)]
        n_scorable = len(scorable_rows)
        n_correct = sum(1 for r in scorable_rows if truthy(r.get("correct", "")))
        n = n_scorable          # every rate below is over the scorable set

        # --- §5.5 / §6.3: code-generation baseline execution outcomes -------
        # "error" is the pre-taxonomy status; syntax_error/runtime_error
        # replaced it. Both are counted so historical cells still aggregate.
        statuses = [str(r.get("exec_status", "")).strip() for r in group]
        exec_ok = sum(1 for s in statuses if s == "ok")
        exec_syntax = sum(1 for s in statuses if s == "syntax_error")
        exec_runtime = sum(1 for s in statuses if s in ("runtime_error", "error"))
        exec_timeout = sum(1 for s in statuses if s == "timeout")
        exec_error = exec_syntax + exec_runtime
        has_exec = any(s for s in statuses)

        # --- §6.4: contract revision and flip rates -------------------------
        # `revised` is None/blank for arms with no revision step, which is
        # different from False (had one, kept the original). Only rows that
        # actually ran a revision step are counted, so the denominator is the
        # number of checked examples rather than the whole cell.
        # The result schema writes contract_checked / contract_revised. This
        # read the legacy `revised` column, which the current writer does not
        # emit, so every contract statistic silently came out as zero on
        # post-refactor rows -- a null result that looked like "the check never
        # fires" rather than like a broken reader.
        #
        # `contract_checked` is explicit: a check that ran and kept its answer
        # is different from an arm that has no check at all, and inferring the
        # distinction from whether a revision field is non-empty conflates
        # them. Legacy rows without the column fall back to `revised`.
        def _checked(r):
            explicit = r.get("contract_checked")
            if not is_blank(explicit):
                return truthy(explicit)
            return not is_blank(r.get("revised"))

        def _revised(r):
            value = r.get("contract_revised")
            if is_blank(value):
                value = r.get("revised")
            return truthy(value)

        checked = [r for r in group if _checked(r)]
        n_revised = sum(1 for r in checked if _revised(r))
        flip_to_correct = flip_to_incorrect = 0
        for r in checked:
            if not _revised(r):
                continue
            before = r.get("pre_revision_correct")
            if is_blank(before):
                continue
            after = truthy(r.get("correct", ""))
            if truthy(before) and not after:
                flip_to_incorrect += 1
            elif not truthy(before) and after:
                flip_to_correct += 1

        # --- §6.5: type adherence -------------------------------------------
        # A non-empty type_warnings list means the value did not inhabit the
        # declared return type. Counted separately from correctness, which is
        # scored on value: a well-typed wrong answer and a wrongly-typed one
        # both score 0, and only this column tells them apart.
        type_warned = 0
        for r in group:
            raw = r.get("type_warnings")
            if is_blank(raw):
                continue
            try:
                parsed = json.loads(raw)
                if parsed:
                    type_warned += 1
            except (ValueError, TypeError):
                type_warned += 1  # non-JSON but non-empty: treat as a warning

        row = {
            "model": model,
            "benchmark": bench,
            "strategy": strat,
            "n": n_scorable,
            "attempted": attempted,
            "scorable": n_scorable,
            "correct": n_correct,
            # Named to say what it is: correct / scorable, not correct /
            # attempted. Consumers that plot "accuracy" get the capability
            # figure, and the infrastructure rate sits beside it.
            "accuracy": round(n_correct / n_scorable, 6) if n_scorable else "",
            "accuracy_over_scorable": (
                round(n_correct / n_scorable, 6) if n_scorable else ""),
            "infrastructure_failures": len(infra_rows),
            "infrastructure_failure_rate": (
                round(len(infra_rows) / attempted, 6) if attempted else ""),
            "type_warned": type_warned,
            "type_adherence": round(1 - type_warned / n, 6) if n else "",
            "n_checked": len(checked),
            "n_revised": n_revised,
            "revision_rate": round(n_revised / len(checked), 6) if checked else "",
            "flip_to_correct": flip_to_correct,
            "flip_to_incorrect": flip_to_incorrect,
            "exec_ok": exec_ok if has_exec else "",
            "exec_error": exec_error if has_exec else "",
            "exec_syntax_error": exec_syntax if has_exec else "",
            "exec_runtime_error": exec_runtime if has_exec else "",
            "exec_timeout": exec_timeout if has_exec else "",
            "exec_failure_rate": (
                round((exec_error + exec_timeout) / n, 6) if has_exec and n else ""
            ),
        }
        out.append(row)
    return out


COLUMNS = [
    "model", "benchmark", "strategy", "n", "attempted", "scorable",
    "correct", "accuracy", "accuracy_over_scorable",
    "infrastructure_failures", "infrastructure_failure_rate",
    "type_warned", "type_adherence",
    "n_checked", "n_revised", "revision_rate",
    "flip_to_correct", "flip_to_incorrect",
    "exec_syntax_error", "exec_runtime_error",
    "exec_ok", "exec_error", "exec_timeout", "exec_failure_rate",
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--raw", type=Path, default=Path("results/eval_raw"),
                    help="ONE directory of per-example result CSVs "
                         "(e.g. results/eval_raw).")
    ap.add_argument("--out", type=Path,
                    default=Path("results/aggregated/summary.csv"),
                    help="Where to write the aggregated long-format CSV.")
    args = ap.parse_args()

    if not args.raw.is_dir():
        print(f"Not a directory: {args.raw}", file=sys.stderr)
        return 1

    rows, problems = load_examples(args.raw)
    for p in problems:
        print(f"  note: {p}", file=sys.stderr)

    if not rows:
        print(f"No usable per-example rows in {args.raw}. Nothing aggregated.",
              file=sys.stderr)
        return 1

    cells = aggregate(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(cells)

    models = sorted({c["model"] for c in cells})
    benches = sorted({c["benchmark"] for c in cells})
    strats = sorted({c["strategy"] for c in cells})
    print(f"Aggregated {len(rows)} example rows into {len(cells)} cells.")
    print(f"  models     : {', '.join(models)}")
    print(f"  benchmarks : {', '.join(benches)}")
    print(f"  strategies : {', '.join(strats)}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
