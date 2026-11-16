"""Offline end-to-end checks of the two-file workflow and recovery semantics."""
import contextlib
import csv
from decimal import Decimal
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError
from evaluation import common
from evaluation import runner
from evaluation.config import configure, read_options


class EvaluationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        lock_temp = tempfile.TemporaryDirectory()
        self.addCleanup(lock_temp.cleanup)
        lock_temp_patch = patch("evaluation.runner.tempfile.gettempdir", return_value=lock_temp.name)
        lock_temp_patch.start()
        self.addCleanup(lock_temp_patch.stop)
        self.release = self.directory / "dataset.json"
        from evaluation.benchmarks.values import encode_value
        benches = {}
        for benchmark in ("humaneval_plus", "mbpp_plus", "gsm8k"):
            items = []
            for i in range(3):
                math = benchmark == "gsm8k"
                inputs = {"question": f"You have {20+i} coins and find 22 more. How many now?"} if math else {"x": 20+i, "y": 22}
                items.append(dict(item_id=f"{benchmark}/{i}", review_status="approved", review_note="Independent arithmetic fixture",
                    evidence=["20+i+22"], public_example_input_sha256=[], signature="solve_math(question: str) -> float" if math else "add(x: int, y: int) -> int",
                    specification="Solve the word problem." if math else "Return the sum of x and y.",
                    scoring_profile="numeric_abs_1e-3" if math else "structured_abs_1e-6" if benchmark=="humaneval_plus" else "mbpp_exact_or_float_v1",
                    observations=[dict(invocation_id="one", inputs=encode_value(inputs), expected=encode_value(float(42+i) if math else 42+i))]))
            benches[benchmark] = dict(source_count=len(items),items=items)
        self.release.write_text(json.dumps(dict(format_version=1,release_id="unit",status="approved",
            policy=dict(public_examples="retained_actual_input_excluded",model_outputs_used_for_curation=False),benchmarks=benches)))
        self.base = self.directory / "result"
        self.calls = []
        self.behavior = None
        self.truncate = False
        self.sdk = SimpleNamespace(responses=SimpleNamespace(create=self.complete),
            chat=SimpleNamespace(completions=SimpleNamespace(create=self.complete)),close=lambda:None)
        self.key_patch = patch.object(common,"read_key",return_value="FAKE_CREDENTIAL_SENTINEL")
        self.sdk_patch = patch.object(common,"build_sdk_client",return_value=self.sdk)
        self.key = self.key_patch.start();self.client = self.sdk_patch.start()
        self.addCleanup(self.key_patch.stop);self.addCleanup(self.sdk_patch.stop)
        sleep_patch = patch("evaluation.backend.time.sleep")
        self.sleep = sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def complete(self,**options):
        self.calls.append(options)
        if self.behavior:
            action = self.behavior(options)
            if isinstance(action,BaseException):raise action
        prompt = options.get("input",options.get("messages"))[-1]["content"]
        value = 43 if "21 coins" in prompt or "'x': 21" in prompt else 44 if "22 coins" in prompt or "'x': 22" in prompt else 42
        if "YOUR PYTHON IMPLEMENTATION:" in prompt:
            content = f"def solve_math(question: str) -> float:\n    return float({value-22} + 22)" if "Signature: solve_math" in prompt else "def add(x: int, y: int) -> int:\n    return x + y"
        else:
            answer = {"verdict":"accept","result":value} if "You are checking a proposed result" in prompt else {"result":value}
            content = "BEGIN FINAL ANSWER\n"+json.dumps(answer)+"\nEND FINAL ANSWER"
        finish = "length" if self.truncate else "stop"
        raw = dict(id="fake",model=options["model"],service_tier="default",choices=[dict(finish_reason=finish,message=dict(role="assistant",content=content))],
            usage=dict(prompt_tokens=20,completion_tokens=50,total_tokens=70,prompt_tokens_details=dict(cached_tokens=0,cache_write_tokens=0),completion_tokens_details=dict(reasoning_tokens=40)))
        if "input" in options:
            raw=dict(id="fake",model=options["model"],service_tier="default",status="completed",output=[dict(type="message",content=[dict(type="output_text",text=content)])],
                usage=dict(input_tokens=20,output_tokens=50,total_tokens=70,input_tokens_details=dict(cached_tokens=0,cache_write_tokens=0),output_tokens_details=dict(reasoning_tokens=40)))
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,message=SimpleNamespace(content=content,tool_calls=None,function_call=None))],
            model_dump=lambda:raw,_request_id="req-unit")

    def invoke(self,*extra,fresh=True):
        argv=["--output",str(self.base)]
        if fresh:
            argv += ["--dataset-release",str(self.release),"--samples","1","--benchmarks","gsm8k","--strategies","direct","--budget","2"]
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            return runner.main(argv+list(extra))

    def document(self):return json.loads(Path(str(self.base)+".json").read_text())

    def test_all_methods_one_row_each_and_full_details_in_both_files(self):
        self.assertEqual(self.invoke("--benchmarks","humaneval_plus","mbpp_plus","gsm8k","--strategies","direct","cot","pseudocode","contract","code_execute","--reasoning","max"),0)
        doc=self.document()
        self.assertEqual(doc["summary"]["correct"],15)
        self.assertEqual(len(self.calls),18)
        self.assertEqual(Decimal(doc["summary"]["reported_usage_cost_usd"]),Decimal("0.000486"))
        self.assertEqual(set(p.name for p in self.directory.iterdir()),{"dataset.json","result.csv","result.json"})
        self.assertTrue(all(c["reasoning"]=={"effort":"max"} and "temperature" not in c and "tools" not in c for c in self.calls))
        self.assertTrue(all(c["store"] is False and c["input"] and c["service_tier"]=="default" for c in self.calls))
        with Path(str(self.base)+".csv").open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),15)
        for row in rows:
            history=json.loads(row["request_history"])
            self.assertTrue(history[0]["dispatched_messages"])
            self.assertTrue(history[-1]["raw_response"])
            self.assertTrue(json.loads(row["attempt_history"])[-1]["row"]["raw_text"])
        self.assertTrue(all(r["generated_code"] for r in rows if r["strategy"]=="code_execute"))
        self.assertTrue(all(r["validation_status"] == "not_checked" for r in rows))
        for row in rows:
            self.assertTrue(row["provider_finish_reason"])
            if row["strategy"] == "contract":
                self.assertEqual(row["verdict_parse_status"], "parsed")
                self.assertEqual(row["verdict_parse_error"], "")
                self.assertTrue(row["verdict_finish_reason"])
            else:
                self.assertEqual((row["verdict_parse_status"], row["verdict_parse_error"],
                                  row["verdict_finish_reason"]), ("", "", ""))
            evaluation = next(item for item in doc["evaluations"] if item["id"] == row["evaluation_id"])
            stored = evaluation["attempts"][-1]["row"]
            for field in ("provider_finish_reason", "verdict_parse_status", "verdict_parse_error", "verdict_finish_reason"):
                self.assertEqual(stored[field], row[field])

    def test_unexpected_error_records_location_without_exception_text(self):
        with patch.object(runner, "evaluate", side_effect=KeyError("SECRET_SENTINEL")):
            self.assertEqual(self.invoke(), 2)
        doc = self.document()
        self.assertEqual((doc["status"], doc["error_type"]), ("failed", "KeyError"))
        self.assertTrue(doc["error_traceback"])
        failed_session = doc["sessions"][-1]
        self.assertEqual(failed_session["error_type"], doc["error_type"])
        self.assertEqual(failed_session["error_traceback"], doc["error_traceback"])
        self.assertNotIn("SECRET_SENTINEL", Path(str(self.base)+".json").read_text())

        self.assertEqual(self.invoke("--resume", fresh=False), 0)
        doc = self.document()
        self.assertNotIn("error_type", doc)
        self.assertNotIn("error_traceback", doc)
        self.assertEqual(doc["sessions"][0], failed_session)
        self.assertNotIn("error_type", doc["sessions"][-1])
        with Path(str(self.base) + ".csv").open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        csv_session = json.loads(row["session_history"])[0]
        for field in ("id", "status", "error_type", "error_traceback"):
            self.assertEqual(csv_session[field], failed_session[field])
        self.assertNotIn("SECRET_SENTINEL", json.dumps(doc))
        self.assertNotIn("SECRET_SENTINEL", json.dumps(row))

    def test_csv_replacement_retries_a_temporary_permission_conflict(self):
        original_replace = common.os.replace
        conflicts = []

        def replace(source, target):
            if target.suffix == ".csv" and not conflicts:
                conflicts.append(target)
                raise PermissionError("temporary CSV sharing conflict")
            original_replace(source, target)

        with patch.object(common.os, "replace", side_effect=replace):
            self.assertEqual(self.invoke(), 0)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(self.document()["summary"]["correct"], 1)
        with Path(str(self.base) + ".csv").open(newline="", encoding="utf-8") as stream:
            self.assertEqual(next(csv.DictReader(stream))["correct"], "True")
        self.assertEqual(set(path.name for path in self.directory.iterdir()),
                         {"dataset.json", "result.csv", "result.json"})

    def test_failed_dispatch_checkpoint_is_not_charged_and_resumes_normally(self):
        original_atomic_json = common.atomic_json
        for exception in (PermissionError("SECRET_SENTINEL"), KeyboardInterrupt()):
            with self.subTest(exception=type(exception).__name__):
                self.base = self.directory / type(exception).__name__
                self.calls.clear()
                failure = [exception]

                def checkpoint(path, doc, *, failure=failure):
                    if failure and doc["requests"] and doc["requests"][-1].get("sdk_dispatched") and not self.calls:
                        raise failure.pop()
                    original_atomic_json(path, doc)

                with patch.object(common, "atomic_json", side_effect=checkpoint):
                    self.assertEqual(self.invoke(), 2)
                doc = self.document()
                self.assertFalse(self.calls)
                self.assertEqual(doc["summary"]["model_calls"], 0)
                self.assertEqual(Decimal(doc["summary"]["conservative_cost_usd"]), 0)
                self.assertEqual(doc["requests"][0]["status"], "interrupted")
                self.assertFalse(doc["requests"][0]["sdk_dispatched"])
                self.assertNotIn("SECRET_SENTINEL", json.dumps(doc))

                self.assertEqual(self.invoke("--resume", fresh=False), 0)
                doc = self.document()
                self.assertEqual(len(self.calls), 1)
                self.assertEqual(doc["summary"]["model_calls"], 1)
                self.assertEqual(doc["summary"]["correct"], 1)
                self.assertEqual(doc["summary"]["conservative_cost_usd"],
                                 doc["summary"]["reported_usage_cost_usd"])
                self.assertFalse(doc["requests"][0]["sdk_dispatched"])
                self.assertTrue(doc["requests"][1]["sdk_dispatched"])

    def test_client_cleanup_failure_keeps_completed_results_and_finalizes_the_session(self):
        self.sdk.close = Mock(side_effect=RuntimeError("CLEANUP_SECRET_SENTINEL"))
        self.assertEqual(self.invoke(), 2)
        doc = self.document()
        self.assertEqual(doc["status"], "completed")
        self.assertEqual(doc["summary"]["correct"], 1)
        session = doc["sessions"][-1]
        self.assertEqual(session["cleanup_error_type"], "RuntimeError")
        self.assertTrue(session["finished_at_utc"])
        self.assertTrue(session["cleanup_error_traceback"])
        with Path(str(self.base) + ".csv").open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["correct"], "True")
        self.assertEqual(json.loads(row["session_history"])[-1]["cleanup_error_type"], "RuntimeError")
        self.assertNotIn("CLEANUP_SECRET_SENTINEL", json.dumps(doc))
        self.assertNotIn("CLEANUP_SECRET_SENTINEL", json.dumps(row))

    def test_cleanup_failure_preserves_the_primary_error_and_checkpoint(self):
        self.behavior = lambda _: KeyError("PRIMARY_SECRET_SENTINEL")
        self.sdk.close = Mock(side_effect=RuntimeError("CLEANUP_SECRET_SENTINEL"))
        self.assertEqual(self.invoke(), 2)
        doc = self.document()
        self.assertEqual((doc["status"], doc["error_type"]), ("failed", "KeyError"))
        self.assertEqual(doc["sessions"][-1]["cleanup_error_type"], "RuntimeError")
        self.assertTrue(doc["sessions"][-1]["finished_at_utc"])
        self.assertEqual(doc["requests"][-1]["status"], "interrupted")
        self.assertNotIn("PRIMARY_SECRET_SENTINEL", json.dumps(doc))
        self.assertNotIn("CLEANUP_SECRET_SENTINEL", json.dumps(doc))

    def test_backend_setup_failure_is_recorded_and_finalized_without_dispatch(self):
        with patch("evaluation.backend.make_backend", side_effect=ImportError("SETUP_SECRET_SENTINEL")):
            self.assertEqual(self.invoke(), 2)
        doc = self.document()
        self.assertEqual((doc["status"], doc["error_type"]), ("failed", "ImportError"))
        self.assertTrue(doc["sessions"][-1]["finished_at_utc"])
        self.assertFalse(self.calls)
        self.assertNotIn("SETUP_SECRET_SENTINEL", json.dumps(doc))

    def test_unknown_retry_label_is_rejected_before_reading_credentials_or_writing_results(self):
        with self.assertRaises(SystemExit) as raised:
            self.invoke("--retry-failed", "timout", fresh=False)
        self.assertEqual(raised.exception.code, 2)
        self.key.assert_not_called()
        self.client.assert_not_called()
        self.assertFalse(Path(str(self.base) + ".json").exists())
        self.assertFalse(Path(str(self.base) + ".csv").exists())

    def test_codegen_execution_failure_is_scored_and_recorded_without_regeneration(self):
        def source_response(**options):
            response = self.complete(**options)
            raw = response.model_dump()
            source = "def solve_math(question): return 1 / 0"
            raw["choices"][0]["message"]["content"] = source
            response.choices[0].message.content = source
            return response
        self.sdk.chat.completions.create = source_response
        self.assertEqual(self.invoke("--strategies", "code_execute", "--execution-timeout", "3", "--execution-retries", "2"), 0)
        doc = self.document()
        attempt = doc["evaluations"][0]["attempts"][-1]
        row = attempt["row"]
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(doc["summary"]["scorable"], 1)
        self.assertEqual(doc["summary"]["correct"], 0)
        self.assertEqual(row["failure_type"], "code_runtime_error")
        self.assertEqual(row["validation_status"], "execution_failed")
        self.assertEqual(row["answer_format"], "python_code_v1")
        self.assertEqual(attempt["generated_code"], "def solve_math(question): return 1 / 0")
        self.assertEqual(len(attempt["execution"]["execution_attempts"]), 1)
        self.assertEqual(attempt["execution"]["error_message"], "division by zero")
        self.assertEqual(attempt["execution"]["error_line"], 1)
        with Path(str(self.base) + ".csv").open(newline="", encoding="utf-8") as stream:
            stored = next(csv.DictReader(stream))
        self.assertEqual(stored["execution_status"], "runtime_error")
        self.assertEqual(stored["generated_code"], attempt["generated_code"])
        details = json.loads(stored["execution_details"])
        self.assertEqual(details["error_message"], "division by zero")
        self.assertEqual(details["error_line"], 1)
        self.assertEqual(details["execution_attempts"][0]["error_message"], "division by zero")
        self.assertEqual(details["execution_attempts"][0]["error_line"], 1)
        self.assertIn("division by zero", stored["execution_error"])
        self.assertEqual((stored["parse_error"], stored["parse_status"]), ("", ""))

    def test_worker_harness_error_is_saved_with_accounting_before_stopping(self):
        details = dict(status="harness_error", value=None, error_type="OSError",
                       error_message="[WinError 5] AssignProcessToJobObject failed",
                       error_line=None, elapsed_seconds=0.01)
        with patch("aicomputing.code_executor.execute_code", return_value=details):
            self.assertEqual(self.invoke("--strategies", "code_execute", "--samples", "2"), 2)
        doc = self.document()
        self.assertEqual(doc["status"], "stopped")
        self.assertTrue(doc["stop_reason"].startswith("worker_harness_failure"))
        self.assertNotIn("error_traceback", doc)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(doc["summary"]["scorable"], 0)
        self.assertEqual(doc["summary"]["infrastructure_excluded"], 1)
        self.assertGreater(Decimal(doc["summary"]["reported_usage_cost_usd"]), 0)
        attempt = doc["evaluations"][0]["attempts"][-1]
        row = attempt["row"]
        self.assertEqual(attempt["status"], "completed")
        self.assertEqual(attempt["execution"]["status"], "harness_error")
        self.assertTrue(attempt["generated_code"])
        self.assertEqual(row["raw_text"], attempt["generated_code"])
        self.assertEqual(row["failure_type"], "harness_error")
        self.assertFalse(row["scorable"])
        self.assertEqual(row["answer_format"], "python_code_v1")
        self.assertEqual((row["input_tokens"], row["output_tokens"]), (20, 50))
        self.assertEqual(doc["summary"]["model_calls"], 1)
        self.assertIn("WinError 5", row["execution_error"])
        with Path(str(self.base) + ".csv").open(newline="", encoding="utf-8") as stream:
            stored = next(csv.DictReader(stream))
        self.assertEqual(stored["failure_type"], "harness_error")
        self.assertEqual(stored["execution_status"], "harness_error")
        self.assertEqual(stored["generated_code"], attempt["generated_code"])
        self.assertIn("WinError 5", json.loads(stored["execution_details"])["error_message"])
        self.assertTrue(json.loads(stored["request_history"])[0]["raw_response"])

    def test_saved_worker_harness_error_can_be_explicitly_retried(self):
        details = dict(status="harness_error", value=None, error_type="OSError",
                       error_message="Worker setup failed", error_line=None, elapsed_seconds=0.01)
        with patch("aicomputing.code_executor.execute_code", return_value=details):
            self.assertEqual(self.invoke("--strategies", "code_execute"), 2)
        self.assertEqual(self.invoke("--retry-failed", "harness_error", fresh=False), 0)
        doc = self.document()
        attempts = doc["evaluations"][0]["attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0]["row"]["failure_type"], "harness_error")
        self.assertTrue(attempts[1]["row"]["correct"])
        self.assertEqual(doc["status"], "completed")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(doc["summary"]["correct"], 1)
        self.assertEqual(doc["summary"]["infrastructure_excluded"], 0)

    def test_malformed_unicode_codegen_reply_is_saved_and_the_run_continues(self):
        source = 'def solve_math(question): return "' + chr(0xD800) + '"'

        def source_response(**options):
            response = self.complete(**options)
            if len(self.calls) == 1:
                response.model_dump()["choices"][0]["message"]["content"] = source
                response.choices[0].message.content = source
            return response

        self.sdk.chat.completions.create = source_response
        self.assertEqual(self.invoke("--strategies", "code_execute", "--samples", "2"), 0)
        doc = self.document()
        self.assertEqual(doc["status"], "completed")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(doc["summary"]["scorable"], 2)
        self.assertEqual(doc["summary"]["correct"], 1)
        self.assertEqual(doc["summary"]["infrastructure_excluded"], 0)
        attempt = doc["evaluations"][0]["attempts"][-1]
        self.assertEqual(attempt["row"]["failure_type"], "code_syntax_error")
        self.assertEqual(attempt["row"]["raw_text"], source)
        self.assertEqual(attempt["generated_code"], source)
        with Path(str(self.base) + ".csv").open(newline="", encoding="utf-8") as stream:
            stored = next(csv.DictReader(stream))
        self.assertIn("\\ud800", stored["raw_text"])
        self.assertEqual(json.loads(stored["request_history"])[0]["response_text"], source)
        self.assertEqual(json.loads(stored["attempt_history"])[-1]["generated_code"], source)

    def test_malformed_numeric_result_is_preserved_and_scored_without_repair(self):
        raw_text = 'BEGIN FINAL ANSWER\n{"result": 42?}\nEND FINAL ANSWER'
        def malformed_response(**options):
            response = self.complete(**options)
            raw = response.model_dump()
            raw["choices"][0]["message"]["content"] = raw_text
            response.choices[0].message.content = raw_text
            return response
        self.sdk.chat.completions.create = malformed_response
        self.assertEqual(self.invoke(), 0)
        row = self.document()["evaluations"][0]["attempts"][-1]["row"]
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(json.loads(row["predicted"]), "42?")
        self.assertEqual(row["parse_status"], "fallback_parsed")
        self.assertEqual(row["failure_type"], "structural_validation_failure")
        self.assertEqual(row["raw_text"], raw_text)
        self.assertFalse(row["correct"])

    def test_interrupted_codegen_retry_keeps_source_and_execution_from_the_scored_attempt(self):
        source = "def solve_math(question): return -1"
        def source_response(**options):
            response = self.complete(**options)
            response.model_dump()["choices"][0]["message"]["content"] = source
            response.choices[0].message.content = source
            return response
        self.sdk.chat.completions.create = source_response
        self.assertEqual(self.invoke("--strategies", "code_execute"), 0)
        first = self.document()["evaluations"][0]["attempts"][0]
        source = "def solve_math(question): return -2"
        with patch("aicomputing.strategies.code_execute.execute_code_with_retries", side_effect=KeyboardInterrupt):
            self.assertEqual(self.invoke("--retry-failed", "all", fresh=False), 2)
        doc = self.document()
        self.assertEqual(doc["evaluations"][0]["attempts"][-1]["generated_code"], source)
        with Path(str(self.base) + ".csv").open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["status"], "interrupted")
        self.assertEqual(row["generated_code"], first["generated_code"])
        self.assertEqual(row["execution_status"], first["execution"]["status"])
        self.assertEqual(json.loads(row["execution_details"]), first["execution"])

    def test_retry_timeouts_keeps_unknown_charges_and_attempt_history(self):
        self.behavior=lambda _:APITimeoutError(request=httpx.Request("POST","https://api.openai.com/v1/responses"))
        self.assertEqual(self.invoke("--retries","1"),0)
        doc=self.document();self.assertEqual(len(self.calls),2)
        self.assertEqual(doc["summary"]["scorable"],0)
        self.assertIsNone(doc["summary"]["accuracy"])
        old=Decimal(doc["summary"]["conservative_cost_usd"])
        self.behavior=None
        self.assertEqual(self.invoke("--retry-failed","timeout","--timeout","900","--retries","0",fresh=False),0)
        doc=self.document();self.assertEqual(doc["summary"]["correct"],1)
        self.assertEqual(len(doc["evaluations"][0]["attempts"]),2)
        self.assertEqual(len(doc["requests"]),3)
        self.assertEqual(Decimal(doc["summary"]["unknown_usage_reservations_usd"]),old)
        self.assertEqual(doc["requests"][-1]["http_phase_timeout_seconds"],900)
        self.assertEqual(self.invoke("--resume",fresh=False),0)
        self.assertEqual(len(self.calls),3)

    def test_temporary_api_failures_retry_and_preserve_both_calls_and_costs(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        errors = [APIConnectionError(request=request), APITimeoutError(request=request)]
        errors += [APIStatusError("ERROR_BODY_SENTINEL", response=httpx.Response(status, request=request),
                                  body={"message": "ERROR_BODY_SENTINEL"})
                   for status in (408, 409, 429, 500, 502, 503, 504)]
        for index, error in enumerate(errors):
            with self.subTest(error=type(error).__name__, status=getattr(error, "status_code", None)):
                self.base = self.directory / f"transient-{index}"
                self.calls.clear()
                self.sleep.reset_mock()
                self.behavior = lambda _, failure=error: failure if len(self.calls) == 1 else None
                self.assertEqual(self.invoke("--retries", "1"), 0)
                doc = self.document()
                self.assertEqual(doc["summary"]["correct"], 1)
                self.assertEqual(len(self.calls), 2)
                first, last = doc["requests"]
                self.assertEqual((first["request_attempt"], last["request_attempt"]), (1, 2))
                self.assertEqual(first["logical_request_first_attempt"], last["logical_request_first_attempt"])
                self.assertNotEqual(first["client_request_id"], last["client_request_id"])
                self.assertTrue(first["retryable_error"])
                self.assertTrue(first["retry_scheduled"])
                self.assertEqual(first["retry_delay_seconds"], 1)
                self.sleep.assert_called_once_with(1)
                self.assertEqual(Decimal(doc["summary"]["unknown_usage_reservations_usd"]), Decimal(first["reservation_usd"]))
                self.assertEqual(Decimal(doc["summary"]["conservative_cost_usd"]),
                                 Decimal(first["charge_usd"]) + Decimal(last["charge_usd"]))
                self.assertNotIn("ERROR_BODY_SENTINEL", json.dumps(doc))
                with Path(str(self.base) + ".csv").open(newline="", encoding="utf-8") as stream:
                    row = next(csv.DictReader(stream))
                self.assertEqual(len(json.loads(row["request_history"])), 2)
                self.assertEqual(row["retries"], "1")

    def test_exhausted_temporary_failures_continue_to_the_next_evaluation(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        self.behavior = lambda _: APIStatusError("unavailable", response=httpx.Response(503, request=request), body=None)
        self.assertEqual(self.invoke("--samples", "2", "--retries", "2"), 0)
        doc = self.document()
        self.assertEqual(doc["status"], "completed")
        self.assertEqual(doc["summary"]["infrastructure_excluded"], 2)
        self.assertEqual(doc["summary"]["scorable"], 0)
        self.assertEqual(len(self.calls), 6)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [1, 2, 1, 2])
        self.assertEqual([call["retry_scheduled"] for call in doc["requests"]], [True, True, False] * 2)
        self.assertEqual([call["retry_delay_seconds"] for call in doc["requests"]], [1, 2, 0] * 2)

    def test_zero_retries_and_backoff_cap_honor_the_requested_attempt_limit(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        self.behavior = lambda _: APIConnectionError(request=request)
        self.assertEqual(self.invoke("--retries", "0"), 0)
        self.assertEqual(len(self.calls), 1)
        self.sleep.assert_not_called()
        self.assertFalse(self.document()["requests"][0]["retry_scheduled"])
        self.assertEqual(self.invoke("--retry-failed", "provider_error", "--retries", "5", fresh=False), 0)
        self.assertEqual(len(self.calls), 7)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [1, 2, 4, 8, 8])

    def test_interrupting_retry_delay_keeps_the_failed_call_for_resume(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        self.behavior = lambda _: APIConnectionError(request=request)
        self.sleep.side_effect = KeyboardInterrupt
        self.assertEqual(self.invoke(), 2)
        doc = self.document()
        self.assertEqual(doc["status"], "interrupted")
        self.assertEqual(len(doc["requests"]), 1)
        self.assertEqual(doc["requests"][0]["status"], "completed")
        charge = Decimal(doc["requests"][0]["charge_usd"])
        self.behavior = None
        self.sleep.side_effect = None
        self.assertEqual(self.invoke("--resume", fresh=False), 0)
        doc = self.document()
        self.assertEqual(doc["summary"]["correct"], 1)
        self.assertEqual(len(doc["requests"]), 2)
        self.assertEqual(Decimal(doc["summary"]["unknown_usage_reservations_usd"]), charge)

    def test_permanent_api_errors_stop_without_retrying_or_attempting_other_tasks(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        for status in (400, 401, 403, 404, 422):
            with self.subTest(status=status):
                self.base = self.directory / f"permanent-{status}"
                self.calls.clear()
                self.behavior = lambda _, code=status: APIStatusError(
                    "ERROR_BODY_SENTINEL", response=httpx.Response(code, request=request), body=None)
                self.assertEqual(self.invoke("--samples", "2", "--retries", "2"), 2)
                doc = self.document()
                self.assertEqual(doc["status"], "stopped")
                self.assertEqual(len(self.calls), 1)
                self.assertFalse(doc["requests"][0]["retryable_error"])
                self.assertFalse(doc["requests"][0]["retry_scheduled"])
                self.sleep.assert_not_called()

    def test_interrupted_run_resumes_without_dataset_and_skips_completed_evaluations(self):
        self.behavior=lambda _:KeyboardInterrupt() if len(self.calls)==2 else None
        self.assertEqual(self.invoke("--samples","2"),2)
        doc=self.document();self.assertEqual(doc["status"],"interrupted")
        self.assertEqual(doc["summary"]["completed_evaluations"],1)
        self.assertEqual(doc["requests"][-1]["status"],"interrupted")
        self.assertTrue(doc["requests"][-1]["sdk_dispatched"])
        self.assertEqual(doc["requests"][-1]["charge_usd"], doc["requests"][-1]["reservation_usd"])
        self.assertGreater(Decimal(doc["summary"]["unknown_usage_reservations_usd"]), 0)
        self.release.unlink();self.behavior=None
        self.assertEqual(self.invoke("--resume",fresh=False),0)
        doc=self.document();self.assertEqual(doc["summary"]["correct"],2)
        self.assertEqual(len(self.calls),3)
        self.assertEqual(len(doc["evaluations"][0]["attempts"]),1)
        self.assertEqual(len(doc["evaluations"][1]["attempts"]),2)
        self.assertEqual(set(p.suffix for p in self.directory.iterdir()),{".csv",".json"})

    def test_budget_stops_before_dispatch_and_can_be_increased_on_resume(self):
        self.assertEqual(self.invoke("--budget","0.000001"),2)
        self.assertFalse(self.calls)
        self.assertEqual(self.document()["status"],"stopped")
        self.assertEqual(self.invoke("--resume","--budget","2",fresh=False),0)
        self.assertEqual(len(self.calls),1)

    def test_config_cli_override_and_dry_run_create_no_results(self):
        config=self.directory/"config.json"
        config.write_text(json.dumps(dict(output="configured",dataset_release="dataset.json",benchmarks=["gsm8k"],strategies=["direct"],samples=3)))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(["--config",str(config),"--samples","1","--dry-run"]),0)
        self.key.assert_not_called();self.assertFalse(self.calls)
        self.assertFalse((self.directory/"configured.csv").exists())
        args,opts=read_options(["--config",str(config),"--samples","1"])
        self.assertEqual(opts["samples"],1);self.assertEqual(args.output,self.directory/"configured")

    def test_malformed_config_returns_a_clear_error_before_credentials_or_result_files(self):
        config = self.directory / "invalid.json"
        config.write_text(json.dumps(dict(output=str(self.base), samples=None)))
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(["--config", str(config)]), 2)
        self.assertIn("samples must be an integer", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())
        self.key.assert_not_called()
        self.client.assert_not_called()
        self.assertFalse(Path(str(self.base) + ".csv").exists())
        self.assertFalse(Path(str(self.base) + ".json").exists())

    def test_changed_condition_is_rejected_and_stale_csv_is_rebuilt(self):
        self.assertEqual(self.invoke(),0)
        self.assertEqual(self.invoke("--resume","--seed","7",fresh=False),2)
        self.assertEqual(len(self.calls),1)
        Path(str(self.base)+".csv").unlink()
        self.assertEqual(self.invoke("--resume",fresh=False),0)
        self.assertTrue(Path(str(self.base)+".csv").exists())

    def test_resume_and_retry_require_original_sources_despite_extraneous_fingerprints(self):
        self.truncate = True
        self.assertEqual(self.invoke(), 0)
        self.truncate = False
        doc = self.document()
        original = dict(doc["provenance"]["source_sha256"])
        json_path, csv_path = Path(str(self.base) + ".json"), Path(str(self.base) + ".csv")
        self.calls.clear()
        self.key.reset_mock()
        self.client.reset_mock()
        for path in ("aicomputing/src/aicomputing/runtime.py", "evaluation/runner.py", "aicomputing/pyproject.toml"):
            changed = dict(original)
            changed[path] = "0" * 64
            doc["provenance"]["alternate_source_sha256"] = changed
            json_path.write_text(json.dumps(doc), encoding="utf-8")
            before_json, before_csv = json_path.read_bytes(), csv_path.read_bytes()
            with patch.object(common, "source_hashes", return_value=changed):
                for action in (("--resume",), ("--retry-failed", "all"), ("--resume", "--dry-run")):
                    with self.subTest(path=path, action=action):
                        self.assertEqual(self.invoke(*action, fresh=False), 2)
                        self.assertFalse(self.calls)
                        self.key.assert_not_called()
                        self.client.assert_not_called()
                        self.assertEqual(json_path.read_bytes(), before_json)
                        self.assertEqual(csv_path.read_bytes(), before_csv)

    def test_extra_fingerprint_does_not_replace_matching_source_hashes(self):
        self.truncate = True
        self.assertEqual(self.invoke(), 0)
        self.truncate = False
        doc = self.document()
        original = dict(doc["provenance"]["source_sha256"])
        doc["provenance"]["alternate_source_sha256"] = {"unexpected-source.py": "0" * 64}
        Path(str(self.base) + ".json").write_text(json.dumps(doc), encoding="utf-8")
        self.assertEqual(self.invoke("--retry-failed", "all", "--timeout", "300", "--retries", "2",
                                     "--execution-timeout", "180", "--execution-retries", "1",
                                     "--budget", "3", fresh=False), 0)
        updated = self.document()
        self.assertEqual(updated["provenance"]["source_sha256"], original)
        self.assertEqual(updated["summary"]["correct"], 1)
        self.assertEqual(len(updated["evaluations"][0]["attempts"]), 2)
        self.assertEqual(len(self.calls), 2)
        for name, value in (("timeout", 300), ("retries", 2), ("execution_timeout", 180),
                            ("execution_retries", 1), ("budget", 3)):
            self.assertEqual(updated["settings"][name], value)

    def test_seeded_samples_are_nested_and_shared_across_models(self):
        first,profile=configure(dict(dataset_release=str(self.release),samples=1,benchmarks=["gsm8k"],strategies=["direct"]))
        second,_=configure(dict(dataset_release=str(self.release),samples=2,benchmarks=["gsm8k"],strategies=["cot"],model="gpt-6.1-terra"))
        _,one=runner.select_tasks(first);_,two=runner.select_tasks(second)
        self.assertEqual(one[0],two[0])
        self.assertEqual(len(one),1);self.assertEqual(len(two),2)

    def test_output_lock_rejects_concurrent_writers(self):
        with runner.output_lock(self.base):
            self.assertEqual(self.invoke(),2)
        self.assertFalse(self.calls)

    def test_parameter_forwarding_and_no_implicit_token_retry(self):
        self.truncate = True
        self.assertEqual(self.invoke("--temperature","0.4","--top-p","0.9","--max-output-tokens","128"),0)
        self.assertEqual(self.calls[0]["temperature"],0.4)
        self.assertEqual(self.calls[0]["top_p"],0.9)
        self.assertEqual(self.calls[0]["max_tokens"],128)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.document()["summary"]["scorable"],1)
        self.assertEqual(self.document()["summary"]["correct"],0)

    def test_authentication_failure_is_recorded_without_secret_error_body(self):
        request=httpx.Request("POST","https://api.openai.com/v1/responses")
        response=httpx.Response(401,request=request)
        self.behavior=lambda _:AuthenticationError("FAKE_CREDENTIAL_SENTINEL",response=response,body={"message":"FAKE_CREDENTIAL_SENTINEL"})
        self.assertEqual(self.invoke("--samples","2"),2)
        doc=self.document();self.assertEqual(len(self.calls),1)
        self.assertEqual(doc["summary"]["infrastructure_excluded"],1)
        self.assertEqual((doc["requests"][0]["http_status"],doc["requests"][0]["http_status_code"]),(401,401))
        self.assertNotIn("FAKE_CREDENTIAL_SENTINEL",Path(str(self.base)+".json").read_text())


class ConfigurationTests(unittest.TestCase):
    def test_malformed_setting_types_raise_configuration_errors(self):
        cases = [dict(samples=None), dict(seed=None), dict(retries=None),
                 dict(model="custom", provider=[])]
        for name in ("samples", "seed", "retries", "execution_retries"):
            cases.extend({name: value} for value in (True, "1", 1.5))
        for name in ("dataset_release", "base_url"):
            cases.extend({name: value} for value in (False, 1, [], {"value": "path"}, "", "  "))
        cases.extend(dict(base_url=value) for value in ("https://example.com:bad/v1",
                     "https://example.com:70000/v1", "https://[invalid/v1"))
        for case in cases:
            with self.subTest(settings=case), self.assertRaises(ValueError):
                configure(case)

    def test_optional_null_settings_still_use_their_documented_defaults(self):
        settings, _ = configure(dict(provider=None, dataset_release=None, base_url=None,
            timeout=None, execution_timeout=None, execution_retries=None, max_output_tokens=None,
            provider_output_tokens=None, temperature=None, top_p=None, input_price=None))
        self.assertEqual((settings["timeout"], settings["execution_timeout"]), (180, 180))
        self.assertEqual(settings["execution_retries"], 1)
        self.assertEqual((settings["temperature"], settings["top_p"]), (0, 1))
        self.assertIsNone(settings["max_output_tokens"])

    def test_malformed_config_paths_can_be_overridden_by_command_line_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps(dict(output=[], key_file={}, dataset_release=5)))
            args, settings = read_options(["--config", str(config), "--output", str(root / "run"),
                "--key-file", str(root / "keys.txt"), "--dataset-release", str(root / "data.json")])
            self.assertEqual(args.output, root / "run")
            self.assertEqual(args.key_file, root / "keys.txt")
            self.assertEqual(settings["dataset_release"], str(root / "data.json"))

    def test_path_options_reject_non_paths_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            for name in ("output", "key_file", "dataset_release"):
                for value in (False, 5, ["path"], {"path": "value"}, "", "\x00"):
                    with self.subTest(name=name, value=value):
                        config.write_text(json.dumps({"output": "run", name: value}))
                        with self.assertRaisesRegex(ValueError, name + " must be a nonempty path"):
                            read_options(["--config", str(config)])

    def test_invalid_cli_paths_are_rejected_before_file_access_or_execution(self):
        for name in ("config", "output", "key_file", "dataset_release"):
            for value in ("", " \t ", "\x00"):
                with self.subTest(name=name, value=value):
                    errors = io.StringIO()
                    with contextlib.redirect_stderr(errors), contextlib.redirect_stdout(io.StringIO()), \
                            patch("evaluation.runner.run") as start, patch.object(Path, "read_text") as read:
                        self.assertEqual(runner.main(["--output", "unused", "--" + name.replace("_", "-"), value]), 2)
                    self.assertIn(name + " must be a nonempty path", errors.getvalue())
                    self.assertNotIn("Traceback", errors.getvalue())
                    read.assert_not_called()
                    start.assert_not_called()

    def test_config_paths_and_cli_paths_keep_their_own_base_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps(dict(output="run", key_file="keys.txt", dataset_release="data.json")))
            args, settings = read_options(["--config", str(config)])
            self.assertEqual(args.config, config.resolve())
            self.assertEqual(args.output, root / "run")
            self.assertEqual(args.key_file, root / "keys.txt")
            self.assertEqual(settings["dataset_release"], str(root / "data.json"))
            args, _ = read_options(["--config", str(config), "--output", "elsewhere/run"])
            self.assertEqual(args.output, Path.cwd() / "elsewhere" / "run")

    def test_retry_selector_accepts_default_explicit_labels_and_all(self):
        for options, expected in ((["--retry-failed"], []),
                                  (["--retry-failed", "timeout", "code_syntax_error"], ["timeout", "code_syntax_error"]),
                                  (["--retry-failed", "all"], ["all"])):
            with self.subTest(options=options):
                args, _ = read_options(["--output", "unused", *options])
                self.assertEqual(args.retry_failed, expected)

    def test_native_reasoning_rejects_sampling_and_local_url_credentials(self):
        for opts in [dict(reasoning="max",temperature=0.0),dict(base_url="https://key:secret@example.com/v1"),dict(samples=0),dict(retries=-1),dict(input_price=float('nan'))]:
            with self.assertRaises(ValueError):configure(opts)

    def test_runs_refuse_another_installed_package_copy(self):
        import aicomputing
        runner.check_package_source()
        foreign = str(Path(tempfile.gettempdir()) / "aicomputing" / "__init__.py")
        for module_path in (foreign, None):
            with self.subTest(module_path=module_path), patch.object(aicomputing, "__file__", module_path):
                with self.assertRaisesRegex(ValueError, "pip install -e"):
                    runner.check_package_source()


if __name__=="__main__":unittest.main()
