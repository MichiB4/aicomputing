"""Trace reviewed MBPP+ observations to published literals, offline.

Creates a new release without resampling, changing labels, or consulting model
responses. Unmatched/conflicting observations are quarantined. Tasks without
precomputed lists retain their explicitly attributed predecessor observations.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "aicomputing/src"), str(ROOT / "aicomputing/benchmarks")]

from evalplus_data import _mbpp_signature, _param_names, _to_kwargs
from mbpp_ground_truth import decode_value, encode_value
from mbpp_published import extract_pairs, NoPublishedTargets, base_input_keys, pool_for, scoring_profile
from prepared_data import input_digest, load_release, validate_release


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def index_source(rows):
    indexed, exclusions = {}, []
    for row in rows:
        item_id = f"Mbpp/{row['task_id']}"
        if item_id in indexed:
            raise ValueError("Duplicate published task: " + item_id)
        signature, entry_point = _mbpp_signature(row["code"])
        names = _param_names(signature)
        task = {"row": row, "signature": signature, "by_input": defaultdict(list)}
        try:
            pairs = extract_pairs(row["test"], entry_point, len(names))
        except NoPublishedTargets as exc:
            task["unavailable"] = str(exc)
        else:
            base = base_input_keys(row["test_list"], entry_point)
            for index, values, expected in pairs.pairs:
                observation = {"invocation_id": f"published:{index}",
                               "pool": pool_for(values, base),
                               "inputs": encode_value(_to_kwargs(values, names)),
                               "expected": encode_value(expected)}
                task["by_input"][input_digest(_to_kwargs(values, names))].append(observation)
            exclusions.extend({"item_id": item_id, "published_index": index, "reason": reason}
                              for index, reason in pairs.excluded.items())
        indexed[item_id] = task
    return indexed, exclusions


def migrate(previous, indexed, *, release_id, source_ref, predecessor_ref):
    validate_release(previous)
    if not release_id.strip() or release_id == previous["release_id"]:
        raise ValueError("A distinct new release ID is required")
    data = copy.deepcopy(previous)
    inventory = data["benchmarks"]["mbpp_plus"]
    if set(indexed) != {item["item_id"] for item in inventory["items"]}:
        raise ValueError("Published task inventory differs from the reviewed release")
    decisions, counts = [], Counter()
    for item in inventory["items"]:
        item_id = item["item_id"]
        task = indexed[item_id]
        if item["review_status"] != "approved":
            decisions.append({"item_id": item_id, "status": "previous_quarantine_retained"})
            counts["previous_quarantine_retained"] += 1
            continue
        row = task["row"]
        # Confirm that inherited reviews describe this exact public task.
        specification = item["specification"]
        original = specification.split("BEGIN ORIGINAL PUBLIC TASK SPECIFICATION\n", 1)
        if (task["signature"] != item["signature"] or len(original) != 2
                or original[1].split("\nEND ORIGINAL PUBLIC TASK SPECIFICATION", 1)[0] != row["prompt"].strip()
                or any(example not in specification for example in row["test_list"])):
            raise ValueError("Published interface/specification/examples changed: " + item_id)
        if scoring_profile(row["test"]) != item["scoring_profile"]:
            raise ValueError("Published scoring profile changed: " + item_id)
        updated, reason = [], None
        for old in item["observations"]:
            if "unavailable" in task:
                new = copy.deepcopy(old)
                new["source"] = {"kind": "reviewed_predecessor_fallback", **predecessor_ref,
                                 "reason": task["unavailable"]}
            else:
                matches = task["by_input"].get(input_digest(decode_value(old["inputs"])), [])
                if not matches:
                    reason = "Reviewed input has no supported published literal pair."
                    break
                # Require exact typed targets, even where the scorer is tolerant.
                if any(match["expected"] != old["expected"] for match in matches):
                    reason = "Published duplicate/target differs from the reviewed typed target."
                    break
                new = copy.deepcopy(matches[0])
                # Preserve the reviewed named input's encoding and order.
                new["inputs"] = old["inputs"]
                new["pool"] = old.get("pool", "unknown")
                new["source"] = {"kind": "published_literals", **source_ref,
                                 "pool_source": "reviewed_predecessor",
                                 "predecessor_invocation_id": old["invocation_id"],
                                 "matching_published_ids": [match["invocation_id"] for match in matches]}
            updated.append(new)
        if reason:
            item.update(review_status="quarantined", reason=reason, observations=[])
            status = "new_quarantine"
        else:
            item["observations"] = updated
            status = ("reviewed_predecessor_fallback" if "unavailable" in task else "published_exact")
        counts[status] += 1
        decisions.append({"item_id": item_id, "status": status,
                          "reason": reason or task.get("unavailable", ""),
                          "observations": [value["invocation_id"] for value in updated] if not reason else []})
    data.update(release_id=release_id, created_at_utc=datetime.now(timezone.utc).isoformat())
    data["policy"]["invocation_freeze"] = (
        "Inherit previously reviewed frozen inputs without resampling. Prefer exact typed published "
        "targets; quarantine mismatches. Prior candidate counts describe the predecessor review, "
        "not newly approved published candidates. Explicit predecessor fallback only when no results list exists.")
    data["provenance"]["mbpp_published_migration"] = {
        "source": source_ref, "predecessor": predecessor_ref,
        "model_outputs_used": False, "reference_execution": False,
        "task_text_changed": False, "new_candidates_approved": False,
    }
    validate_release(data)
    return data, {"decisions": decisions, "counts": dict(counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-arrow", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--previous-release", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.out_dir.resolve()
    if output.exists():
        raise FileExistsError("Refusing to replace an existing release directory")
    previous, predecessor = load_release(args.previous_release)
    source_raw = args.source_arrow.read_bytes()
    import pyarrow as pa
    rows = pa.ipc.open_stream(pa.BufferReader(source_raw)).read_all().to_pylist()
    indexed, exclusions = index_source(rows)
    source_ref = {"dataset": "evalplus/mbppplus", "split": "test", "revision": args.source_revision,
                  "sha256": digest(source_raw), "file": "mbppplus-test.arrow"}
    predecessor_ref = {"release_id": predecessor["release_id"], "sha256": predecessor["sha256"],
                       "file": "previous-release.json"}
    data, report = migrate(previous, indexed, release_id=args.release_id,
                           source_ref=source_ref, predecessor_ref=predecessor_ref)
    report.update(source=source_ref, predecessor=predecessor_ref,
                  extraction={"source_tasks": len(indexed),
                              "tasks_with_supported_pairs": sum(bool(t["by_input"]) for t in indexed.values()),
                              "supported_pairs": sum(sum(map(len, t["by_input"].values())) for t in indexed.values()),
                              "excluded_pairs": len(exclusions),
                              "without_precomputed_results": [k for k, t in indexed.items() if "unavailable" in t]},
                  model_calls=0, reference_execution=False)
    release_raw = (json.dumps(data, ensure_ascii=True, indent=2) + "\n").encode("utf-8")
    report["release_sha256"] = digest(release_raw)
    report["preparation_source_sha256"] = {
        path.relative_to(ROOT).as_posix(): digest(path.read_bytes())
        for path in [Path(__file__), ROOT / "aicomputing/benchmarks/mbpp_published.py",
                     ROOT / "aicomputing/benchmarks/evalplus_data.py",
                     ROOT / "aicomputing/benchmarks/mbpp_ground_truth.py",
                     ROOT / "aicomputing/benchmarks/prepared_data.py"]}
    predecessor_raw = args.previous_release.read_bytes()
    if digest(predecessor_raw) != predecessor_ref["sha256"] or args.source_arrow.read_bytes() != source_raw:
        raise ValueError("Input changed during preparation")
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {"dataset-release.json": release_raw, "mbppplus-test.arrow": source_raw,
                 "previous-release.json": predecessor_raw,
                 "comparison.json": (json.dumps(report, indent=2) + "\n").encode("utf-8")}
    for name, raw in artifacts.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
    with (output / "pair-exclusions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item_id", "published_index", "reason"])
        writer.writeheader()
        writer.writerows(exclusions)
    print(json.dumps({"output": str(output), "counts": report["counts"],
                      "extraction": report["extraction"],
                      "release_sha256": report["release_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
