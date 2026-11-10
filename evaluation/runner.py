"""Run evaluations and maintain the CSV results and JSON checkpoint."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from decimal import Decimal
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback
import uuid
from importlib.metadata import PackageNotFoundError, version

from evaluation import common
from evaluation.config import configure, find_release, read_options

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = ROOT / "aicomputing" / "src"
SCHEMA = 1


def check_package_source():
    """Refuse to run against another copy of aicomputing than the hashed source."""
    import aicomputing
    module_path = getattr(aicomputing, "__file__", None)
    if module_path is None or not Path(module_path).resolve().is_relative_to(PACKAGE_SOURCE):
        raise ValueError(
            "Evaluation requires this checkout's aicomputing package. "
            'From the repository root, run: python -m pip install -e "./aicomputing[openai,benchmarks]"'
        )


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


@contextmanager
def output_lock(base):
    """Coordinate writers through a persistent file outside the results folder."""
    identity = os.path.normcase(str(Path(base).resolve()))
    filename = hashlib.sha256(os.fsencode(identity)).hexdigest() + ".lock"
    path = Path(tempfile.gettempdir()) / "aicomputing-evaluation-locks" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never unlink: waiting writers must keep referring to the same file.
    # Closing the stream releases the OS lock, including when the body raises.
    with path.open("a+b") as stream:
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("Another evaluation is using this --output") from None
        if not os.fstat(stream.fileno()).st_size:
            stream.write(b"0")
            stream.flush()
        yield


def select_tasks(settings):
    from evaluation.benchmarks import tasks as benchmark_tasks
    from evaluation.benchmarks.values import encode_value
    path = find_release(ROOT, settings["dataset_release"])
    selection, resolved = benchmark_tasks.select_tasks(path, settings["benchmarks"], settings["samples"], settings["seed"])
    tasks = []
    for benchmark, items in resolved.items():
        for item in items:
            data = asdict(item)
            data["inputs"] = encode_value(item.inputs)
            data["expected"] = encode_value(item.expected)
            task = {"benchmark": benchmark, "item": data}
            task["sha256"] = digest(task)
            tasks.append(task)
    return selection, tasks


def decode_task(task):
    from evaluation.benchmarks.tasks import ResolvedItem
    from evaluation.benchmarks.values import decode_value
    data = dict(task["item"])
    data["inputs"] = decode_value(data["inputs"])
    data["expected"] = decode_value(data["expected"])
    return ResolvedItem(**data)


def new_document(settings, profile):
    from aicomputing.strategies.code_prompt import PROMPT_REVISION
    from aicomputing.prompting import FRAMED_PROMPT_REVISION
    selection, tasks = select_tasks(settings)
    dependencies = {}
    for name in ("openai", "httpx", "ollama", "pydantic", "truststore"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    doc = {"format": "aicomputing-evaluation", "schema_version": SCHEMA, "run_id": str(uuid.uuid4()),
        "status": "planned", "created_at_utc": common.utc_now(), "settings": settings,
        "pricing": profile.pricing_snapshot(), "selection": selection, "tasks": tasks,
        "provenance": {"python": platform.python_version(), "dependencies": dependencies,
                       "source_sha256": common.source_hashes(ROOT),
                       "code_prompt_revision": PROMPT_REVISION, "text_prompt_revision": FRAMED_PROMPT_REVISION},
        "sessions": [], "requests": [], "evaluations": []}
    for i in range(len(tasks)):
        for strategy in settings["strategies"]:
            doc["evaluations"].append({"id": str(len(doc["evaluations"]) + 1), "task_index": i,
                                      "strategy": strategy, "attempts": []})
    return doc


def latest_row(evaluation):
    return next((a["row"] for a in reversed(evaluation["attempts"]) if a.get("row") is not None), None)


def summarize(doc):
    rows = [r for e in doc["evaluations"] if (r := latest_row(e)) is not None]
    scored, correct = sum(r["scorable"] for r in rows), sum(r["correct"] for r in rows)
    requests = doc["requests"]
    reported = sum((Decimal(r.get("charge_usd", "0")) for r in requests if r.get("usage_complete")), Decimal(0))
    conservative = sum((Decimal(r.get("charge_usd", "0")) for r in requests), Decimal(0))
    cells = []
    for benchmark in doc["settings"]["benchmarks"]:
        for strategy in doc["settings"]["strategies"]:
            chosen = [r for r in rows if r["benchmark"] == benchmark and r["strategy"] == strategy]
            n = sum(r["scorable"] for r in chosen)
            cells.append(dict(benchmark=benchmark, strategy=strategy, completed=len(chosen), scorable=n,
                correct=sum(r["correct"] for r in chosen), accuracy=sum(r["correct"] for r in chosen) / n if n else None))
    return dict(planned_evaluations=len(doc["evaluations"]), completed_evaluations=len(rows), scorable=scored,
        correct=correct, infrastructure_excluded=len(rows) - scored,
        accuracy=correct / scored if scored else None, correct_over_all_completed=correct / len(rows) if rows else None,
        model_calls=sum(bool(r.get("sdk_dispatched")) for r in requests),
        reported_usage_cost_usd=str(reported), conservative_cost_usd=str(conservative),
        unknown_usage_reservations_usd=str(conservative - reported),
        cells=cells)


def csv_rows(doc):
    from evaluation.benchmarks.values import decode_value
    requests = {}
    for record in doc["requests"]:
        requests.setdefault(record.get("evaluation_id"), []).append(record)
    sessions = {}
    for session in doc["sessions"]:
        details = {key: value for key, value in session.items() if key != "evaluation_ids"}
        for evaluation_id in session["evaluation_ids"]:
            sessions.setdefault(evaluation_id, []).append(details)
    for evaluation in doc["evaluations"]:
        task = doc["tasks"][evaluation["task_index"]]
        item = task["item"]
        row = latest_row(evaluation)
        last = evaluation["attempts"][-1] if evaluation["attempts"] else {}
        selected_attempt = next((a for a in reversed(evaluation["attempts"]) if a.get("row") is not None), last)
        if row is None:
            row = dict(run_id=doc["run_id"], model=doc["settings"]["model"], benchmark=task["benchmark"],
                strategy=evaluation["strategy"], benchmark_item_id=item["item_id"], signature=item["signature"],
                docstring=item["docstring"], inputs=json.dumps(decode_value(item["inputs"]), default=str),
                expected=json.dumps(decode_value(item["expected"]), default=str), predicted="", correct=None, scorable=None)
        calls = requests.get(evaluation["id"], [])
        reported = sum((Decimal(r.get("charge_usd", "0")) for r in calls if r.get("usage_complete")), Decimal(0))
        conservative = sum((Decimal(r.get("charge_usd", "0")) for r in calls), Decimal(0))
        yield {**row, "evaluation_id": evaluation["id"], "status": last.get("status", "pending"),
            "provider": doc["settings"]["provider"], "reasoning_effort": doc["settings"]["reasoning"],
            "generated_code": selected_attempt.get("generated_code", ""),
            "execution_status": selected_attempt.get("execution", {}).get("status", ""),
            "execution_details": selected_attempt.get("execution"), "inputs_typed": item["inputs"], "expected_typed": item["expected"],
            "task_sha256": task["sha256"], "cost_usd_all_attempts": str(conservative),
            "reported_usage_cost_usd_all_attempts": str(reported),
            "unknown_usage_reservations_usd_all_attempts": str(conservative - reported),
            "reasoning_tokens_reported_all_attempts": sum(r.get("reasoning_tokens") or 0 for r in calls),
            "request_history": calls, "attempt_history": evaluation["attempts"],
            "session_history": sessions.get(evaluation["id"], []),
            "run_settings": selected_attempt.get("settings", doc["settings"])}


def save(doc, base, *, csv_output=True):
    doc["updated_at_utc"] = common.utc_now()
    doc["summary"] = summarize(doc)
    common.atomic_json(Path(str(base) + ".json"), doc)
    if not csv_output:
        return
    from evaluation.benchmarks.result_schema import RESULT_COLUMNS
    extras = ["evaluation_id", "status", "provider", "reasoning_effort", "generated_code", "execution_status",
              "execution_details", "inputs_typed", "expected_typed", "task_sha256", "cost_usd_all_attempts",
              "reported_usage_cost_usd_all_attempts", "unknown_usage_reservations_usd_all_attempts",
              "reasoning_tokens_reported_all_attempts",
              "request_history", "attempt_history", "session_history", "run_settings"]
    target = Path(str(base) + ".csv")
    temporary = Path(str(target) + ".tmp")
    with temporary.open("w", encoding="utf-8", errors="backslashreplace", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS + extras)
        writer.writeheader()
        for row in csv_rows(doc):
            writer.writerow({k: json.dumps(v, ensure_ascii=True, default=str) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})
        stream.flush()
        os.fsync(stream.fileno())
    common.replace_file(temporary, target)


def recover_interrupted(doc):
    for call in doc["requests"]:
        if call.get("status") in ("prepared", "in_flight"):
            call.update(status="interrupted", usage_complete=False, finished_at_utc=common.utc_now(),
                        charge_usd=call["reservation_usd"] if call.get("sdk_dispatched") else "0")
    for evaluation in doc["evaluations"]:
        for attempt in evaluation["attempts"]:
            if attempt["status"] == "running":
                attempt.update(status="interrupted", finished_at_utc=common.utc_now())


def evaluate(doc, evaluation, backend, settings, checkpoint):
    from aicomputing import CodeExecutionError, VirtualRuntime
    from aicomputing.request import ExecuteRequest, ExecuteResult, FailureType
    from aicomputing.status import ParseStatus, ValidationStatus
    from aicomputing.strategies import create_strategy
    from aicomputing.strategies.code_prompt import ANSWER_FORMAT as CODE_ANSWER_FORMAT
    from evaluation.backend import InfrastructureFailure, ModelOutputFailure
    from evaluation.benchmarks.prepared_data import code_task_specification
    from evaluation.benchmarks.result_schema import build_row
    from evaluation.benchmarks.scoring import compare
    from evaluation.benchmarks.values import encode_value
    task = doc["tasks"][evaluation["task_index"]]
    item, benchmark, strategy = decode_task(task), task["benchmark"], evaluation["strategy"]
    attempt = dict(id=str(uuid.uuid4()), status="running", started_at_utc=common.utc_now(),
                   session_id=doc["sessions"][-1]["id"], settings=dict(settings), generated_code=None)
    evaluation["attempts"].append(attempt)
    checkpoint()
    ledger = backend.ledger
    ledger.context = dict(evaluation_id=evaluation["id"], evaluation_attempt=attempt["id"],
                          benchmark=benchmark, strategy=strategy, item_id=item.item_id)
    before = len(ledger.records)
    generation = dict(model=settings["model"], temperature=settings["temperature"] or 0.0,
                      top_p=1.0 if settings["top_p"] is None else settings["top_p"], max_new_tokens=settings["max_output_tokens"])
    execution = None
    stop_reason = None
    try:
        selected_strategy, docstring = strategy, item.docstring
        if strategy == "code_execute":
            docstring = code_task_specification(item.docstring)
            def record_code(code):
                attempt["generated_code"] = code
                checkpoint()
            selected_strategy = create_strategy(
                strategy, execution_timeout=settings["execution_timeout"],
                execution_retries=settings["execution_retries"],
                task_mode="word_problem" if benchmark == "gsm8k" else "function",
                on_generated_code=record_code,
            )
        with VirtualRuntime(backend=backend, model=settings["model"], strategy=selected_strategy, validation_mode="none") as runtime:
            result = runtime.execute(ExecuteRequest(signature=item.signature, docstring=docstring, inputs=item.inputs,
                strategy=strategy,
                validation_mode="none", max_retries=0, **generation))
        execution = result.execution_details
    except InfrastructureFailure as exc:
        calls = ledger.records[before:]
        result = common.infrastructure_result(calls, model=settings["model"], strategy=strategy)
        result.failure_type = exc.kind
        if exc.fatal:
            stop_reason = "provider_rejected_request; correct the API configuration before continuing"
    except CodeExecutionError as exc:
        calls = ledger.records[before:]
        execution = exc.execution_details
        result = common.infrastructure_result(calls, model=settings["model"], strategy=strategy)
        result.failure_type = FailureType.HARNESS_ERROR
        result.answer_format = CODE_ANSWER_FORMAT
        result.generated_code = exc.generated_code
        result.execution_details = execution
        result.execution_error = "harness_error: " + (execution.get("error_type") or "unknown")
        if execution.get("error_message"):
            result.execution_error += ": " + execution["error_message"]
        raw_response = calls[-1].get("raw_response") or {}
        result.provider_finish_reason = calls[-1].get("finish_reason") or raw_response.get("status")
        stop_reason = "worker_harness_failure; correct worker setup before continuing"
    except ModelOutputFailure as exc:
        calls = ledger.records[before:]
        result = ExecuteResult(result=None, raw_text=calls[-1].get("response_text") or "", model=settings["model"],
            strategy=strategy, validation_status=ValidationStatus.UNPARSED, failure_type=FailureType.PARSE,
            parse_status=ParseStatus.FAILED, parse_error=str(exc), prompts=[m["content"] for r in calls for m in r["dispatched_messages"] if m["role"] == "user"])
    calls = ledger.records[before:]
    accounting = common.account_attempts(result, calls)
    correct = (result.failure_type is None and result.validation_status in (ValidationStatus.VALID, ValidationStatus.NOT_CHECKED)
               and (execution is None or execution["status"] == "ok")
               and compare(result.result, item.expected, item.scoring_profile))
    wire = dict(temperature=settings["temperature"], top_p=settings["top_p"], max_new_tokens=settings["max_output_tokens"])
    row = build_row(result, run_id=doc["run_id"], benchmark=benchmark, item_id=item.item_id, signature=item.signature,
        docstring=item.docstring, inputs=item.inputs, expected=item.expected, correct=bool(correct), invocation_pool=item.invocation_pool,
        return_type_source=item.return_type_source, generation=wire)
    row.update(accounting)
    if execution is not None:
        row["latency_ms"] += round(1000 * execution["elapsed_seconds"])
        row["retries"] += max(0, len(execution.get("execution_attempts", [])) - 1)
    attempt.update(status="completed", finished_at_utc=common.utc_now(), row=row)
    if execution is not None:
        attempt["execution"] = {**execution, "value": encode_value(execution["value"])}
    checkpoint()
    if stop_reason:
        raise common.StopRun(stop_reason)


def run(args, supplied):
    from evaluation.backend import make_backend
    check_package_source()
    output = args.output
    json_path, csv_path = Path(str(output) + ".json"), Path(str(output) + ".csv")
    extending = args.resume or args.retry_failed is not None
    if extending:
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        if doc.get("format") != "aicomputing-evaluation" or doc.get("schema_version") != SCHEMA:
            raise ValueError("--resume/--retry-failed requires a JSON from this command")
        settings, profile = configure(supplied, doc["settings"])
        if doc["provenance"]["source_sha256"] != common.source_hashes(ROOT):
            raise ValueError("Package or evaluation code changed; use the original code to resume or choose a new --output")
        for task in doc["tasks"]:
            if digest({k: v for k, v in task.items() if k != "sha256"}) != task["sha256"]:
                raise ValueError("Saved task content was modified")
        recover_interrupted(doc)
    else:
        if json_path.exists() or csv_path.exists():
            raise ValueError("Output already exists; use --resume, --retry-failed, or another --output")
        settings, profile = configure(supplied)
        doc = new_document(settings, profile)
    if args.retry_failed is not None:
        failures = set(args.retry_failed or ["timeout", "provider_error"])
        selected = [e for e in doc["evaluations"] if (r := latest_row(e)) is not None and not r["correct"]
                    and ("all" in failures or r["failure_type"] in failures)]
    else:
        selected = [e for e in doc["evaluations"] if latest_row(e) is None
                    or (e["attempts"] and e["attempts"][-1]["status"] == "interrupted")]
    if Decimal(str(settings["budget"])) < Decimal(summarize(doc)["conservative_cost_usd"]):
        raise ValueError("Budget is below the already recorded charge; increase --budget")
    print(f"{settings['model']} / reasoning={settings['reasoning']}: {len(selected)} evaluations; "
          f"{settings['samples']} tasks per benchmark; timeout {settings['timeout']:g}s; {settings['retries']} retry/retries.", flush=True)
    if args.dry_run:
        print("Outputs: " + str(csv_path) + " and " + str(json_path))
        return 0
    if not selected:
        save(doc, output)
        print("No matching evaluations to run.")
        return 0
    key = common.read_key(args.key_file, profile.key_env)
    from evaluation.common import configure_system_certificates
    configure_system_certificates()
    doc.update(settings=settings, status="running")
    for field in ("stop_reason", "error_type", "error_traceback"):
        doc.pop(field, None)
    session = dict(id=str(uuid.uuid4()), operation="retry_failed" if args.retry_failed is not None else "resume" if args.resume else "run",
                   started_at_utc=common.utc_now(), settings=dict(settings), evaluation_ids=[e["id"] for e in selected])
    doc["sessions"].append(session)
    save(doc, output)
    def emit(event):
        event = dict(event)
        kind = event.pop("event")
        if kind == "request_start":
            doc["requests"].append({**event, "status": "prepared", "charge_usd": "0", "usage_complete": False})
        else:
            call = next(r for r in reversed(doc["requests"]) if r["attempt"] == event["attempt"])
            if kind == "sdk_dispatch":
                before_dispatch = dict(call)
            call.update(event, status="completed" if kind == "request_complete" else "in_flight")
            if kind == "sdk_dispatch":
                call.update(charge_usd=call["reservation_usd"], usage_complete=False)
        try:
            save(doc, output, csv_output=False)
        except BaseException:
            if kind == "sdk_dispatch":
                # The SDK call follows this checkpoint. If saving fails, it
                # cannot have been sent; finalization must not charge it.
                call.clear()
                call.update(before_dispatch)
            raise
    backend = None
    try:
        backend = make_backend(profile, settings, key, emit, doc["requests"])
        for index, evaluation in enumerate(selected, 1):
            evaluate(doc, evaluation, backend, settings, lambda: save(doc, output))
            row = latest_row(evaluation)
            print(f"[{index}/{len(selected)}] {row['benchmark']}/{row['strategy']} {row['benchmark_item_id']}: "
                  f"{row['failure_type'] or 'correct'}; cost ${backend.ledger.charged:.5f}", flush=True)
        doc["status"] = "completed" if all(latest_row(e) is not None for e in doc["evaluations"]) else "incomplete"
    except KeyboardInterrupt:
        doc["status"] = "interrupted"
    except common.StopRun as exc:
        doc.update(status="stopped", stop_reason=str(exc))
    except Exception as exc:
        # Keep the code location, but never record exception text: SDK bodies may carry credentials.
        error = dict(error_type=type(exc).__name__, error_traceback=traceback.format_tb(exc.__traceback__))
        doc.update(status="failed", **error)
        session.update(error)
        print(f"Evaluation failed with {type(exc).__name__}; see error_traceback in {json_path}", file=sys.stderr)
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                session.update(cleanup_error_type=type(exc).__name__,
                               cleanup_error_traceback=traceback.format_tb(exc.__traceback__))
                print(f"Evaluation client cleanup failed with {type(exc).__name__}; see session history.", file=sys.stderr)
        recover_interrupted(doc)
        session.update(finished_at_utc=common.utc_now(), status=doc["status"])
        if doc.get("stop_reason"):
            session["stop_reason"] = doc["stop_reason"]
        save(doc, output)
    summary = doc["summary"]
    print(f"{doc['status']}: {summary['correct']}/{summary['scorable']} correct; "
          f"{summary['infrastructure_excluded']} infrastructure exclusions; "
          f"reported ${Decimal(summary['reported_usage_cost_usd']):.5f}, conservative ${Decimal(summary['conservative_cost_usd']):.5f}.")
    print(str(csv_path) + "\n" + str(json_path))
    return 0 if doc["status"] == "completed" and "cleanup_error_type" not in session else 2


def main(argv=None):
    try:
        args, supplied = read_options(argv)
        if args.dry_run:
            return run(args, supplied)
        with output_lock(args.output):
            return run(args, supplied)
    except (ValueError, OSError) as exc:
        print("Evaluation configuration error: " + str(exc), file=sys.stderr)
        return 2
