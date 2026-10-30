"""Aggregate current per-evaluation CSV rows without regrading predictions."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

COLUMNS = [
    "model", "benchmark", "strategy", "n", "attempted", "scorable",
    "correct", "accuracy", "accuracy_over_scorable",
    "infrastructure_failures", "infrastructure_failure_rate",
    "type_warned", "type_adherence", "n_checked", "n_revised", "revision_rate",
]
REQUIRED = {"model", "benchmark", "strategy", "correct", "scorable", "contract_status", "type_warnings"}


def truthy(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def load_examples(directory):
    rows, problems = [], []
    previous_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(max(previous_limit, 2**31 - 1))
        for path in sorted(directory.glob("*.csv")):
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not REQUIRED <= set(reader.fieldnames or ()):
                    problems.append(f"{path.name}: missing current result columns; skipped")
                    continue
                rows.extend(reader)
    finally:
        csv.field_size_limit(previous_limit)
    return rows, problems


def aggregate(rows):
    cells = defaultdict(list)
    for row in rows:
        missing = REQUIRED - row.keys()
        if missing:
            raise ValueError("Missing result columns: " + ", ".join(sorted(missing)))
        if row["contract_status"] not in ("not_checked", "unchanged", "revised"):
            raise ValueError("Unknown contract_status")
        cells[row["model"], row["benchmark"], row["strategy"]].append(row)
    output = []
    for (model, benchmark, strategy), group in sorted(cells.items()):
        scorable = [row for row in group if truthy(row["scorable"])]
        count, attempted = len(scorable), len(group)
        correct = sum(truthy(row["correct"]) for row in scorable)
        checked = [row for row in scorable if row["contract_status"] != "not_checked"]
        revised = sum(row["contract_status"] == "revised" for row in checked)
        warned = 0
        for row in scorable:
            raw = row["type_warnings"]
            if raw in (None, "", "None", "null"):
                continue
            try:
                warned += bool(json.loads(raw))
            except (ValueError, TypeError):
                warned += 1
        accuracy = round(correct / count, 6) if count else ""
        output.append({
            "model": model, "benchmark": benchmark, "strategy": strategy,
            "n": count, "attempted": attempted, "scorable": count, "correct": correct,
            "accuracy": accuracy, "accuracy_over_scorable": accuracy,
            "infrastructure_failures": attempted - count,
            "infrastructure_failure_rate": round((attempted - count) / attempted, 6),
            "type_warned": warned, "type_adherence": round(1 - warned / count, 6) if count else "",
            "n_checked": len(checked), "n_revised": revised,
            "revision_rate": round(revised / len(checked), 6) if checked else "",
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Directory containing current result CSVs")
    parser.add_argument("--out", type=Path, help="Defaults to summary.csv inside the input directory")
    args = parser.parse_args()
    if not args.raw.is_dir():
        parser.error("raw must name an existing directory")
    rows, problems = load_examples(args.raw)
    for problem in problems:
        print(problem, file=sys.stderr)
    if not rows:
        print("No current result rows found.", file=sys.stderr)
        return 1
    cells = aggregate(rows)
    output = args.out or args.raw / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(cells)
    print(f"Aggregated {len(rows)} rows into {len(cells)} cells: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
