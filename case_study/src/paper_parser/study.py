"""Paired evaluation on the independently annotated TDMSci sentences."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import platform
import random
import time

from . import conventional, model
from .contract import FIELDS, LABELS, REVISION, SIGNATURE, SPECIFICATION
from .dataset import Case, DATA_DIR, FILES, load_split


def select_cases(split: str, sample_size: int | None, seed: int, ids: list[str] | None):
    cases = load_split(split)
    if ids:
        wanted = set(ids)
        if len(wanted) != len(ids) or wanted - {case.id for case in cases}:
            raise ValueError("Case IDs must be distinct and present in the chosen split")
        cases = [case for case in cases if case.id in wanted]
    if sample_size is not None:
        if not 1 <= sample_size <= len(cases):
            raise ValueError("Sample size must fit the selected cases")
        cases = random.Random(seed).sample(cases, sample_size)
        cases.sort(key=lambda case: case.id)
    return cases


def grade(case: Case, outcome):
    actual = outcome.record.spans() if outcome.status == "completed" and outcome.record else set()
    expected = case.gold.spans()
    fields = {}
    for field in FIELDS:
        label = LABELS[field]
        gold = {span for span in expected if span[0] == label}
        predicted = {span for span in actual if span[0] == label}
        fields[field] = {"true_positive": len(gold & predicted),
                         "false_positive": len(predicted - gold),
                         "false_negative": len(gold - predicted),
                         "exact": outcome.status == "completed" and gold == predicted}
    return {"exact_sentence": outcome.status == "completed" and actual == expected,
            "fields": fields}


def make_record(case: Case, outcome, settings: model.Settings):
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "case_id": case.id, "split": case.split, "text": case.sentence.text,
            "tokens": list(case.sentence.tokens), "token_offsets": list(case.sentence.offsets),
            "gold": case.gold.to_dict(), "outcome": outcome.to_dict(),
            "grading": grade(case, outcome),
            "estimated_api_cost_usd": model.estimated_cost(outcome, settings),
            "settings": settings.public() if outcome.engine == "model"
                        else {"algorithm": "training-phrase-lexicon-v1"},
            "specification_revision": REVISION, "signature": SIGNATURE,
            "specification": SPECIFICATION}


def summary(records):
    answer = {}
    for engine in sorted({row["outcome"]["engine"] for row in records}):
        rows = [row for row in records if row["outcome"]["engine"] == engine]
        per_field = {}
        for field in FIELDS:
            stats = {key: sum(row["grading"]["fields"][field][key] for row in rows)
                     for key in ("true_positive", "false_positive", "false_negative")}
            tp, fp, fn = (stats[key] for key in ("true_positive", "false_positive", "false_negative"))
            per_field[field] = {**stats,
                                "precision": tp / (tp + fp) if tp + fp else None,
                                "recall": tp / (tp + fn) if tp + fn else None,
                                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
        tp = sum(item["true_positive"] for item in per_field.values())
        fp = sum(item["false_positive"] for item in per_field.values())
        fn = sum(item["false_negative"] for item in per_field.values())
        answer[engine] = {"sentences": len(rows),
                          "failed_attempts": sum(row["outcome"]["status"] != "completed" for row in rows),
                          "exact_sentences": sum(row["grading"]["exact_sentence"] for row in rows),
                          "exact_sentence_accuracy": sum(row["grading"]["exact_sentence"] for row in rows) / len(rows),
                          "micro_precision": tp / (tp + fp) if tp + fp else None,
                          "micro_recall": tp / (tp + fn) if tp + fn else None,
                          "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                          "by_field": per_field,
                          "mean_seconds": sum(row["outcome"]["elapsed_seconds"] for row in rows) / len(rows),
                          "known_estimated_api_cost_usd": sum(row["estimated_api_cost_usd"] or 0 for row in rows),
                          "unknown_or_unpriced_cost_attempts": sum(row["estimated_api_cost_usd"] is None for row in rows)}
    return answer


def csv_text(records):
    columns = ["case_id", "split", "text", "tokens", "engine", "status",
               "exact_sentence", "field_grading", "elapsed_seconds", "estimated_api_cost_usd",
               "gold", "predicted", "raw_text", "prompts", "input_tokens", "output_tokens", "details"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in records:
        outcome = row["outcome"]
        prediction = outcome["prediction"] or {}
        flat = {"case_id": row["case_id"], "split": row["split"], "text": row["text"],
                "tokens": row["tokens"], "engine": outcome["engine"], "status": outcome["status"],
                "exact_sentence": row["grading"]["exact_sentence"],
                "field_grading": row["grading"]["fields"],
                "elapsed_seconds": outcome["elapsed_seconds"],
                "estimated_api_cost_usd": row["estimated_api_cost_usd"],
                "gold": row["gold"], "predicted": outcome["record"],
                **{key: prediction.get(key) for key in ("raw_text", "prompts", "input_tokens", "output_tokens")},
                "details": row}
        writer.writerow({key: json.dumps(value, ensure_ascii=True) if isinstance(value, (dict, list))
                         else value for key, value in flat.items()})
    return buffer.getvalue()


def save(document, output: Path):
    document["summary"] = summary(document["records"])
    for suffix, content in ((".json", json.dumps(document, ensure_ascii=True, indent=2) + "\n"),
                            (".csv", csv_text(document["records"]))):
        target = Path(str(output) + suffix)
        temporary = Path(str(target) + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(.05 * 2 ** attempt)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate short-text entity tagging on TDMSci.")
    parser.add_argument("--engine", choices=("rules", "model", "both"), default="rules")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--sample-size", type=int, help="Number of sentences, sampled with --seed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case-id", action="append", help="Select one or more IDs, e.g. T0001")
    parser.add_argument("--output", type=Path, default=Path("output/tdmsci-rules"))
    parser.add_argument("--model", default="gpt-6.1-terra")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--input-price", type=float)
    parser.add_argument("--output-price", type=float)
    args = parser.parse_args(argv)
    try:
        settings = model.Settings(model=args.model, key_file=args.key_file, timeout=args.timeout,
                                  max_output_tokens=args.max_output_tokens,
                                  input_price=args.input_price, output_price=args.output_price)
        if args.engine != "rules" and not settings.has_credentials():
            raise ValueError("Provide --key-file or configure OPENAI_API_KEY")
        cases = select_cases(args.split, args.sample_size, args.seed, args.case_id)
        output = args.output.resolve()
        if output.suffix.lower() in (".csv", ".json"):
            output = output.with_suffix("")
        output.parent.mkdir(parents=True, exist_ok=True)
        lock = Path(str(output) + ".lock")
        lock.touch(exist_ok=False)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    try:
        if any(Path(str(output) + suffix).exists() for suffix in (".csv", ".json")):
            parser.error("Output exists; choose a new --output")
        import aicomputing
        package_root = Path(aicomputing.__file__).parent
        code_root = Path(__file__).parent
        document = {"format": "tdmsci-entity-study", "schema_version": 1,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                    "engine": args.engine, "split": args.split,
                    "selection": {"sample_size": args.sample_size, "seed": args.seed, "case_ids": [case.id for case in cases]},
                    "python": platform.python_version(), "package_version": version("aicomputing"),
                    "specification_revision": REVISION,
                    "settings": settings.public() if args.engine != "rules" else None,
                    "dataset": {name: {"file": file, "sha256": sha}
                                for name, (file, sha) in FILES.items()},
                    "dataset_license_sha256": hashlib.sha256((DATA_DIR / "LICENSE").read_bytes()).hexdigest(),
                    "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in code_root.glob("*.py")},
                    "package_source_sha256": {p.relative_to(package_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in package_root.rglob("*.py")}, "records": []}
        engines = ("rules", "model") if args.engine == "both" else (args.engine,)
        try:
            save(document, output)
            with settings.runtime() if "model" in engines else nullcontext(None) as runtime:
                for case in cases:
                    for engine in engines:
                        outcome = (conventional.extract(case.sentence) if engine == "rules"
                                   else model.extract(case.sentence, runtime, settings))
                        row = make_record(case, outcome, settings)
                        document["records"].append(row)
                        save(document, output)
                        print(f"{case.id} {engine}: {outcome.status}; exact={row['grading']['exact_sentence']}", flush=True)
                        if outcome.status in {"timeout", "provider_error", "provider_unavailable"}:
                            document["status"] = "stopped"
                            return 2
            document["status"] = "completed"
            return 0
        except KeyboardInterrupt:
            document["status"] = "interrupted"
            return 2
        except Exception as exc:
            document.update(status="failed", error_type=type(exc).__name__)
            print(f"Study failed with {type(exc).__name__}; see saved JSON.")
            return 2
        finally:
            document["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            save(document, output)
            print(json.dumps(document["summary"], indent=2))
            print(str(output) + ".csv\n" + str(output) + ".json")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
