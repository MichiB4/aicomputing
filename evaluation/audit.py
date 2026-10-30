"""Read-only consistency audit for completed runs of the maintained workflow."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath

RESULT_COLUMNS_V7 = (
    "run_id",
    "model",
    "strategy",
    "benchmark",
    "benchmark_item_id",
    "signature",
    "docstring",
    "inputs",
    "expected",
    "predicted",
    "correct",
    "scorable",
    "validation_status",
    "structural_validity",
    "failure_type",
    "parse_status",
    "parse_error",
    "contract_status",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "retries",
    "invocation_pool",
    "return_type_source",
    "temperature",
    "top_p",
    "max_new_tokens",
    "type_warnings",
    "raw_text",
    "prompts",
    "reasoning_requests",
    "reasoning_observations",
    "answer_format",
    "transport_diagnostics",
)


class AuditError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise AuditError(code)


def number(value):
    require(type(value) is int and value >= 0, "nonnegative_integer_required")
    return value


def money(value):
    require(isinstance(value, str), "decimal_string_required")
    result = Decimal(value)
    require(result.is_finite() and result >= 0, "finite_nonnegative_amount_required")
    return result


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_artifact(root, relative):
    path = PurePosixPath(relative)
    require(relative and not path.is_absolute() and not set(path.parts) & {".", ".."}
            and "\\" not in relative and ":" not in relative, "unsafe_artifact_path")
    resolved = (root / relative).resolve(strict=True)
    require(resolved.is_relative_to(root), "artifact_escapes_run_directory")
    return resolved.read_bytes()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate_json_field")
        value[key] = item
    return value


def decode(raw):
    def invalid_constant(_):
        raise AuditError("nonfinite_json_constant")
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=invalid_constant)


def snapshots(root, manifest, metadata):
    protocol = manifest["experiment_protocol"]
    versions = {7: "evaluation-luna-text-timeout-v2"}
    version = metadata["runner_schema_version"]
    require(type(version) is int and version in versions
            and metadata["runner_protocol"] == protocol["runner_protocol"] == versions[version],
            "unsupported_run_schema")
    require(metadata["status"] == "completed", "run_not_completed")
    require(digest(read_artifact(root, "manifest.executed.json")) == metadata["manifest_sha256"], "manifest_hash")
    reference = manifest["dataset_release"]
    require(reference["path"] == "dataset-release.json", "nonportable_dataset_reference")
    raw = read_artifact(root, "dataset-release.json")
    require(digest(raw) == reference["sha256"] == metadata["dataset_release_sha256"], "dataset_hash")
    data = decode(raw)
    require(data["release_id"] == reference["release_id"] and data["status"] == "approved", "dataset_release_identity")
    reproducibility = manifest["reproducibility"]
    require(type(reproducibility["format_version"]) is int and reproducibility["format_version"] == 1,
            "reproducibility_format")
    raw = read_artifact(root, "requirements.txt")
    require(digest(raw) == reproducibility["requirements_sha256"] == metadata["requirements_sha256"], "requirements_hash")
    expected = reproducibility["environment"]
    pins = ["# Python " + expected["python"]["version"]]
    pins.extend(name + "==" + version for name, version in sorted(expected["packages"].items()))
    require(raw == ("\n".join(pins) + "\n").encode("utf-8"), "requirements_content")
    actual = metadata["environment"]
    require(actual["python"] == expected["python"] and actual["packages"] and expected["packages"]
            and all(expected["packages"].get(name) == version for name, version in actual["packages"].items()),
            "recorded_software_versions")
    source_pins = protocol["source_sha256"]
    require(source_pins and source_pins == metadata["source_sha256"], "source_inventory")
    archived = {path.relative_to(root / "executed-source").as_posix()
                for path in (root / "executed-source").rglob("*") if path.is_file()}
    require(archived == set(source_pins), "archived_source_coverage")
    for relative, expected_hash in source_pins.items():
        require(digest(read_artifact(root, "executed-source/" + relative)) == expected_hash, "archived_source_hash")
    require(digest(read_artifact(root, "runner.executed.py")) == protocol["runner_sha256"]
            == metadata["runner_sha256"] == source_pins["evaluation/runner.py"], "runner_snapshot")
    require(digest(read_artifact(root, "prompting.executed.py"))
            == source_pins["aicomputing/src/aicomputing/prompting.py"], "prompt_snapshot")
    sizes = [len(entry["items"]) for entry in manifest["benchmarks"].values()]
    require(sizes and len(set(sizes)) == 1 and sizes[0] > 0 and metadata["cases_per_benchmark"] == sizes[0]
            and all(entry["selected"] == entry["requested"] == len(entry["items"])
                    for entry in manifest["benchmarks"].values()), "manifest_case_counts")
    require(metadata["model"] in manifest["models"] and len(manifest["models"]) == 1, "model_identity")
    require(metadata["answer_format"] == manifest["answer_format"] == protocol["answer_format"], "answer_format")
    require(metadata["generation"] == manifest["generation"]
            and metadata["generation"]["max_retries"] == 0 and metadata["generation"]["max_new_tokens"] is None,
            "generation_settings")
    require(metadata["sdk_retries"] == protocol["sdk_retries"] == 0
            and metadata["validation_retries"] == protocol["validation_retries"] == 0, "hidden_retries")
    require(metadata["timeout_retries"] == protocol["timeout_retries"]
            and metadata["http_phase_timeout_seconds"] == protocol["uniform_http_phase_timeout_seconds"], "timeout_policy")
    require(metadata["reasoning_requested"] == {"reasoning_effort": protocol["native_reasoning_effort"]}
            and metadata["tool_policy"]["tools_offered"] is False and metadata["tool_policy"]["execute_tools"] is False
            and protocol["tools_supplied"] is False and protocol["application_output_cap"] is None, "declared_native_policy")
    require(all(metadata["budget"][key] == value for key, value in protocol["pricing"].items()), "price_assumptions")
    return data


def attempt_charge(record, pricing):
    rates = {name: money(pricing[name + "_usd_per_million"]) / 1_000_000
             for name in ("input", "cached_input", "cache_write", "output")}
    bound = len(json.dumps(record["dispatched_messages"], ensure_ascii=True).encode("utf-8")) + number(pricing["framing_reserve"])
    reservation = bound * rates["cache_write"] + number(pricing["provider_output_reservation_tokens"]) * rates["output"]
    require(record["input_reservation_tokens"] == bound <= number(pricing["max_input_reservation_tokens"])
            and record["provider_output_reservation_tokens"] == pricing["provider_output_reservation_tokens"]
            and money(record["reservation_usd"]) == reservation, "attempt_reservation")
    if not record.get("returned_model"):
        require(record["usage_complete"] is False and record.get("input_tokens") is None
                and record.get("output_tokens") is None and record["charge_basis"] == "reservation_unknown_usage",
                "unknown_usage_not_invented")
        charge = reservation
    else:
        inputs, outputs = number(record["input_tokens"]), number(record["output_tokens"])
        cached, written = record.get("cached_input_tokens_reported"), record.get("cache_write_tokens_reported")
        reads = 0 if cached is None else number(cached)
        writes = inputs - reads if written is None else number(written)
        require(0 <= reads + writes <= inputs and writes >= 0 and inputs <= bound
                and outputs <= pricing["provider_output_reservation_tokens"], "reported_usage_bounds")
        upper_bound = cached is None or written is None
        require(record["usage_complete"] is True and record["cached_input_tokens_accounted"] == reads
                and record["cache_write_tokens_accounted"] == writes
                and record["charge_is_upper_bound"] is upper_bound
                and record["charge_basis"] == ("reported_usage_cache_upper_bound" if upper_bound else "reported_usage"),
                "cache_accounting")
        charge = (inputs - reads - writes) * rates["input"] + reads * rates["cached_input"] + writes * rates["cache_write"] + outputs * rates["output"]
    require(money(record["charge_usd"]) == charge and charge <= reservation, "attempt_charge")
    return charge, reservation


def requests(events, manifest, metadata):
    protocol = manifest["experiment_protocol"]
    indexed = {name: {} for name in ("request_start", "sdk_dispatch", "sdk_response", "request_complete", "transport_failure")}
    exclusions = {}
    for position, event in enumerate(events):
        name = event["event"]
        if name == "evaluation_infrastructure_exclusion":
            key = (event["benchmark"], event["strategy"], event["item_id"])
            require(key not in exclusions, "duplicate_infrastructure_exclusion")
            exclusions[key] = event
            continue
        require(name in indexed, "unknown_request_event")
        attempt = number(event["attempt"])
        require(attempt > 0 and attempt not in indexed[name], "duplicate_or_invalid_attempt")
        indexed[name][attempt] = (position, event)
    completed = indexed["request_complete"]
    expected_ids = list(range(1, len(completed) + 1))
    require(list(completed) == list(indexed["request_start"]) == list(indexed["sdk_dispatch"]) == expected_ids,
            "physical_attempt_coverage")
    require(set(indexed["sdk_response"]) <= set(completed)
            and set(indexed["transport_failure"]) <= set(completed), "orphan_response_or_failure")
    totals = {name: Decimal(0) for name in ("conservative_charge_usd", "usage_based_charge_usd",
                                          "exact_reported_charge_usd", "unknown_usage_reservations_usd")}
    limits = protocol["cost_guard"]
    budget, ceiling = money(limits["operational_usd"]), money(limits["user_ceiling_usd"])
    require(0 < budget <= ceiling and all(money(metadata["budget"][name]) == money(value) for name, value in limits.items()),
            "frozen_budget")
    records, groups, identities = [], defaultdict(list), set()
    for attempt in expected_ids:
        finish, record = completed[attempt]
        started, start = indexed["request_start"][attempt]
        dispatched, wire = indexed["sdk_dispatch"][attempt]
        require(started < dispatched < finish and start["dispatched_messages"] == wire["dispatched_messages"] == record["dispatched_messages"],
                "wire_message_provenance")
        require(all(record.get(key) == value for key, value in start.items()
                    if key not in ("event", "sdk_wire_checked", "sdk_dispatched")), "request_context")
        require(type(record["elapsed_seconds"]) in (int, float) and math.isfinite(record["elapsed_seconds"])
                and record["elapsed_seconds"] >= 0, "attempt_elapsed_time")
        number(record["request_attempt"])
        require(record["requested_model"] == metadata["model"] and record["sdk_dispatched"] is True
                and record["sdk_wire_checked"] is True, "request_dispatch")
        require(record["application_max_new_tokens"] is None and wire["sdk_token_limit_keys"] == record["sdk_token_limit_keys"] == []
                and wire["tools_offered"] is record["sdk_tools_offered"] is False and not record.get("tool_use_detected")
                and wire["tool_choice_omitted"] is record["sdk_tool_choice_omitted"] is True, "token_and_tool_policy")
        require(wire["reasoning_effort"] == record["sdk_reasoning_effort"] == protocol["native_reasoning_effort"]
                and wire["service_tier"] == record["sdk_service_tier"] == "default"
                and record.get("service_tier") in (None, "default"), "native_settings")
        require(wire["temperature"] == manifest["generation"]["temperature"]
                and wire["top_p"] == manifest["generation"]["top_p"]
                and wire["http_phase_timeout_seconds"] == record["http_phase_timeout_seconds"]
                == protocol["uniform_http_phase_timeout_seconds"], "wire_generation_settings")
        client_id = wire["client_request_id"]
        require(isinstance(client_id, str) and client_id and client_id not in identities
                and record["client_request_id"] == client_id, "client_request_identity")
        identities.add(client_id)
        returned = bool(record.get("returned_model"))
        require((attempt in indexed["sdk_response"]) == returned
                and (attempt in indexed["transport_failure"]) != returned, "response_and_failure_coverage")
        if returned:
            response_position, response = indexed["sdk_response"][attempt]
            require(dispatched < response_position < finish and record["returned_model"] == metadata["model"]
                    and all(record.get(key) == value for key, value in response.items() if key not in ("event", "attempt")),
                    "response_evidence")
            require(record.get("finish_reason") != "length" and record.get("normalized_finish_reason") != "length"
                    and record.get("exception_type") is None, "completed_response")
        charge, reservation = attempt_charge(record, protocol["pricing"])
        require(totals["conservative_charge_usd"] + reservation <= min(budget, ceiling), "dispatch_budget_reservation")
        totals["conservative_charge_usd"] += charge
        kind = "usage_based_charge_usd" if returned else "unknown_usage_reservations_usd"
        totals[kind] += charge
        if returned and record["charge_basis"] == "reported_usage":
            totals["exact_reported_charge_usd"] += charge
        require(money(record["cumulative_conservative_charge_usd"]) == totals["conservative_charge_usd"], "cumulative_charge")
        groups[number(record["logical_request_first_attempt"])].append(record)
        records.append(record)
    for first, group in groups.items():
        require(first > 0 and [record["attempt"] for record in group] == list(range(first, first + len(group)))
                and [record["request_attempt"] for record in group] == list(range(1, len(group) + 1))
                and len(group) <= number(protocol["timeout_retries"]) + 1, "logical_attempt_sequence")
        for index, record in enumerate(group):
            require(all(record[key] == group[0][key] for key in ("benchmark", "strategy", "item_id", "dispatched_messages")),
                    "logical_request_context")
            retry = index < len(group) - 1
            require(not retry or (record.get("exception_type") == "BackendTimeoutError" and not record.get("returned_model")),
                    "timeout_only_retries")
            if not record.get("returned_model"):
                position, failure = indexed["transport_failure"][record["attempt"]]
                require(position > completed[record["attempt"]][0] and failure["timeout_retry_scheduled"] is retry
                        and record["exception_type"] in ("BackendTimeoutError", "BackendCommunicationError")
                        and failure["exception_type"] == record["exception_type"]
                        and failure["logical_request_first_attempt"] == first and failure["request_attempt"] == record["request_attempt"]
                        and record.get("http_status_code") is None and not any(
                            record.get(name) for name in ("provider_error_code", "provider_error_type", "provider_error_param"))
                        and money(failure["retained_unknown_reservation_usd"]) == money(record["charge_usd"]), "transport_failure_evidence")
    return records, totals, exclusions


def results(root, manifest, data, rows, records, exclusions):
    require(all(set(row) == set(RESULT_COLUMNS_V7) for row in rows), "row_columns")
    expected = {(benchmark, strategy, item["item_id"]): item
                for benchmark, entry in manifest["benchmarks"].items()
                for strategy in manifest["strategies"] for item in entry["items"]}
    require(expected and len(expected) == sum(len(entry["items"]) for entry in manifest["benchmarks"].values()) * len(manifest["strategies"]),
            "distinct_manifest_cases")
    require(Counter((row["benchmark"], row["strategy"], row["benchmark_item_id"]) for row in rows) == Counter(expected.keys()),
            "evaluation_coverage")
    previous_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(max(previous_limit, 2**31 - 1))
        reader = csv.DictReader(io.StringIO(read_artifact(root, "results.csv").decode("utf-8"), newline=""))
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), "csv_columns")
        require(reader.fieldnames == list(RESULT_COLUMNS_V7), "csv_columns")
        csv_rows = list(reader)
    finally:
        csv.field_size_limit(previous_limit)
    require(len(csv_rows) == len(rows) and all(set(csv_row) == set(row) and all(
        csv_row[key] == ("" if value is None else str(value)) for key, value in row.items())
        for csv_row, row in zip(csv_rows, rows)), "csv_jsonl_agreement")
    calls = defaultdict(list)
    for record in records:
        key = (record["benchmark"], record["strategy"], record["item_id"])
        require(key in expected, "attempt_outside_selected_cases")
        calls[key].append(record)
    tasks = {benchmark: {item["item_id"]: item for item in entry["items"] if item["review_status"] == "approved"}
             for benchmark, entry in data["benchmarks"].items()}
    excluded = set()
    for row in rows:
        key = (row["benchmark"], row["strategy"], row["benchmark_item_id"])
        selected, group = expected[key], calls[key]
        task = tasks[key[0]][key[2]]
        require(selected["invocation_id"] in {item["invocation_id"] for item in task["observations"]}
                and row["signature"] == task["signature"] and row["docstring"] == task["specification"], "selected_task_identity")
        require(type(row["correct"]) is bool and type(row["scorable"]) is bool
                and (bool(row["failure_type"]) is (not row["correct"])) and (not row["correct"] or row["scorable"]), "outcome_flags")
        require(row["model"] == manifest["models"][0] and row["answer_format"] == manifest["answer_format"]
                and row["invocation_pool"] == (selected.get("invocation_pool") or "")
                and all(row[name] == manifest["generation"][name] for name in ("temperature", "top_p", "max_new_tokens")), "row_settings")
        require(row["structural_validity"] in ("not_assessed", "valid", "invalid")
                and row["contract_status"] in ("not_checked", "unchanged", "revised"), "row_statuses")
        complete = all(record["usage_complete"] for record in group)
        inputs = sum(record.get("input_tokens") or 0 for record in group)
        outputs = sum(record.get("output_tokens") or 0 for record in group)
        retries = sum(record["request_attempt"] > 1 for record in group)
        require(group and number(row["retries"]) == retries, "row_attempt_accounting")
        require(number(row["latency_ms"]) == round(1000 * sum(record["elapsed_seconds"] for record in group)), "row_latency")
        require(all(row[name] == (value if complete else None) for name, value in (
            ("input_tokens", inputs), ("output_tokens", outputs), ("total_tokens", inputs + outputs))), "row_usage_completeness")
        logical = list(dict.fromkeys(record["logical_request_first_attempt"] for record in group))
        require(len(logical) <= (2 if row["strategy"] == "contract" else 1)
                and (row["strategy"] != "contract" or not row["scorable"] or len(logical) == 2)
                and row["scorable"] is bool(group[-1].get("returned_model")), "strategy_request_count")
        diagnostics = [{**(record.get("transport_diagnostics") or {}), "attempt": record["attempt"],
                        "logical_request_first_attempt": record["logical_request_first_attempt"],
                        "request_attempt": record["request_attempt"], "exception_type": record.get("exception_type")}
                       for record in group]
        observations = [{"reported_reasoning_tokens": record.get("reasoning_tokens"),
                         "returned_reasoning_chars": record.get("returned_reasoning_chars"), "source": "provider_response"}
                        if record.get("returned_model") else None for record in group]
        require(decode(row["transport_diagnostics"]) == diagnostics and decode(row["reasoning_requests"])
                == [{"reasoning_effort": manifest["experiment_protocol"]["native_reasoning_effort"]} for record in group]
                and decode(row["reasoning_observations"]) == observations, "per_attempt_diagnostics")
        if not row["scorable"]:
            excluded.add(key)
            event = exclusions[key]
            require(event["attempts"] == [record["attempt"] for record in group]
                    and event["scorable"] is False and event["retry"] is False and event["complete_usage"] is False
                    and event["partial_reported_input_tokens"] == inputs and event["partial_reported_output_tokens"] == outputs
                    and not group[-1].get("returned_model") and row["failure_type"] == event["failure_type"]
                    == ("timeout" if group[-1]["exception_type"] == "BackendTimeoutError" else "provider_error"),
                    "infrastructure_exclusion")
    require(set(exclusions) == excluded, "infrastructure_exclusion_coverage")


def summary_matches(manifest, metadata, summary, rows, records, totals):
    protocol = manifest["experiment_protocol"]
    planned = sum(len(entry["items"]) for entry in manifest["benchmarks"].values()) * len(manifest["strategies"])
    require(number(protocol["evaluations"]) == number(metadata["planned_evaluations"]) == len(rows) == planned, "planned_evaluations")
    expected = {"completed_evaluations": len(rows), "generation_attempts": len(records),
                "returned_responses": sum(bool(record.get("returned_model")) for record in records)}
    require(len(records) <= number(protocol["max_generation_attempts"]) == metadata["max_generation_attempts"], "attempt_cap")
    for name, value in expected.items():
        require(number(metadata[name]) == number(summary[name]) == value, "summary_counts")
    for name, value in totals.items():
        require(money(metadata[name]) == money(summary[name]) == value, "summary_costs")
    counts = {
        "timeout_retries": sum(record["request_attempt"] > 1 for record in records),
        "infrastructure_excluded_evaluations": sum(not row["scorable"] for row in rows),
        "wire_checks_passed": len(records), "requests_with_missing_usage": sum(not record["usage_complete"] for record in records),
        "tool_responses": 0, "truncated_requests": 0,
    }
    for field in ("input_tokens", "output_tokens", "cached_input_tokens_reported", "cached_input_tokens_accounted",
                  "cache_write_tokens_reported", "cache_write_tokens_accounted", "reasoning_tokens"):
        name = field if field.endswith(("_reported", "_accounted")) else field + "_reported"
        counts[name] = sum(number(record[field]) if record.get(field) is not None else 0 for record in records)
    returned = [record for record in records if record.get("returned_model")]
    counts.update(
        responses_missing_cached_input_count=sum(record.get("cached_input_tokens_reported") is None for record in returned),
        responses_missing_cache_write_count=sum(record.get("cache_write_tokens_reported") is None for record in returned),
        responses_missing_reasoning_usage=sum(record.get("reasoning_tokens") is None for record in returned),
        requests_with_cache_cost_upper_bound=sum(bool(record.get("charge_is_upper_bound")) for record in records))
    require(summary["returned_models"] == sorted({record["returned_model"] for record in returned}), "returned_model_summary")
    for name, value in counts.items():
        require(number(summary[name]) == value, "summary_usage_and_retries")
    require(summary["status"] == metadata["status"] == "completed" and not summary["stop_reason"]
            and summary["wall_seconds"] == metadata["wall_seconds"]
            and type(metadata["wall_seconds"]) in (int, float) and math.isfinite(metadata["wall_seconds"])
            and metadata["wall_seconds"] >= 0, "completed_summary")
    cells = {(cell["benchmark"], cell["strategy"]): cell for cell in summary["cells"]}
    require(len(cells) == len(summary["cells"]) == len(manifest["benchmarks"]) * len(manifest["strategies"]), "summary_cells")
    for benchmark in manifest["benchmarks"]:
        for strategy in manifest["strategies"]:
            selected = [row for row in rows if row["benchmark"] == benchmark and row["strategy"] == strategy]
            calls = [record for record in records if record["benchmark"] == benchmark and record["strategy"] == strategy]
            cell = cells[(benchmark, strategy)]
            expected_cell = {"completed": len(selected), "scorable": sum(row["scorable"] for row in selected),
                             "correct": sum(row["correct"] for row in selected),
                             "validation_failures": sum(row["validation_status"] != "valid" for row in selected),
                             "generation_attempts": len(calls), "timeout_retries": sum(record["request_attempt"] > 1 for record in calls)}
            require(all(number(cell[name]) == value for name, value in expected_cell.items())
                    and money(cell["charge_usd"]) == sum((money(record["charge_usd"]) for record in calls), Decimal(0)),
                    "cell_totals")


def audit_run(path):
    """Check archived records without importing or executing their source code."""
    passed = []
    try:
        root = Path(path).resolve(strict=True)
        manifest = decode(read_artifact(root, "manifest.executed.json"))
        metadata = decode(read_artifact(root, "metadata.json"))
        summary = decode(read_artifact(root, "summary.json"))
        data = snapshots(root, manifest, metadata)
        passed.append("snapshots")
        events = [decode(line) for line in read_artifact(root, "requests.jsonl").splitlines()]
        rows = [decode(line)["row"] for line in read_artifact(root, "evaluations.jsonl").splitlines()]
        records, totals, exclusions = requests(events, manifest, metadata)
        passed.append("requests")
        results(root, manifest, data, rows, records, exclusions)
        passed.append("results")
        summary_matches(manifest, metadata, summary, rows, records, totals)
        passed.append("summary")
        scorable, correct = sum(row["scorable"] for row in rows), sum(row["correct"] for row in rows)
        return {"status": "PASS", "scope": "record_consistency", "passed_stages": passed,
                "evaluations": len(rows), "scorable": scorable, "correct": correct,
                "accuracy": correct / scorable if scorable else None, "model_calls": len(records),
                "accounting": {name: str(value) for name, value in totals.items()}}
    except (OSError, ValueError, KeyError, TypeError, IndexError, csv.Error, InvalidOperation) as exc:
        return {"status": "FAIL", "scope": "record_consistency", "passed_stages": passed,
                "failed_check": str(exc) if isinstance(exc, AuditError) else "artifact_format_" + type(exc).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit_run(args.run_dir)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
