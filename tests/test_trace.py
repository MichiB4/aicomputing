"""Trace invocation results without changing prediction behavior."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from aicomputing import Backend, Constraint, ExecuteRequest, ExecuteResult, Tracer, VirtualRuntime
from aicomputing.answer_protocol import frame_answer
from aicomputing.backends.base import GenerateResponse


class TraceBackend(Backend):
    name = "offline"
    default_model = "offline"

    def __init__(self, *responses):
        self.responses = responses or (frame_answer('{"result": 5}'),)
        self.calls = []

    def cache_state(self):
        return {"responses": self.responses}

    def generate(self, prompt, **settings):
        text = self.responses[min(len(self.calls), len(self.responses) - 1)]
        self.calls.append(prompt)
        return GenerateResponse(text=text, model="offline", input_tokens=2, output_tokens=3,
                                requested_reasoning={"reasoning_effort": "none"})


def request(**settings):
    return ExecuteRequest(signature="f(x: int) -> int", docstring="Return x.", inputs={"x": 5},
                          max_retries=0, **settings)


def result(**settings):
    defaults = dict(result=5, raw_text=frame_answer('{"result": 5}'), model="offline", strategy="direct",
                    validation_status="valid", parse_status="parsed", input_tokens=2, output_tokens=3,
                    total_tokens=5, model_calls=1, latency_ms=12.345)
    return ExecuteResult(**(defaults | settings))


class TraceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = self.directory / "nested" / "trace.jsonl"

    def entries(self):
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def test_none_explicitly_disables_runtime_tracing(self):
        backend = TraceBackend()
        with VirtualRuntime(backend=backend, trace_file=None) as runtime:
            self.assertIsNone(runtime.tracer)
            self.assertEqual(runtime.execute(request()).result, 5)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_string_and_path_trace_destinations_are_preserved_and_written(self):
        destinations = (str(self.directory / " string trace.jsonl"), self.directory / " path trace.jsonl")
        for destination in destinations:
            with self.subTest(destination=destination):
                backend = TraceBackend()
                with VirtualRuntime(backend=backend, trace_file=destination) as runtime:
                    self.assertEqual(runtime.tracer.path, Path(destination))
                    self.assertEqual(runtime.execute(request()).result, 5)
                entry = json.loads(Path(destination).read_text(encoding="utf-8"))
                self.assertEqual(entry["result"], 5)

    def test_direct_logging_records_returned_value_diagnostics_and_exact_raw_text(self):
        raw = '  <think>draft</think>\nBEGIN FINAL ANSWER\n{"result": 5}\nEND FINAL ANSWER  \n'
        snapshot = result(raw_text=raw)
        tracer = Tracer(self.path)
        tracer.log(request(), snapshot)
        entry, = self.entries()
        self.assertEqual(entry["inputs"], {"x": 5})
        self.assertEqual((entry["result"], entry["parsed_result"], entry["parsed_type"]), (5, 5, "int"))
        self.assertEqual(entry["raw_text"], raw)
        self.assertEqual(entry["model_calls"], 1)
        self.assertEqual(entry["parse_status"], "parsed")
        self.assertIsNone(entry["parse_error"])
        self.assertIsNone(entry["execution_error"])
        self.assertIsNone(entry["failure_type"])
        self.assertEqual(entry["latency_ms"], 12.3)
        self.assertEqual(tracer.call_count, 1)

    def test_retry_totals_and_cached_invocation_are_separate_from_final_attempt_evidence(self):
        final_text = frame_answer('{"result": 5}')
        backend = TraceBackend(frame_answer('{"result": "wrong type"}'), final_text)
        call = request()
        call.max_retries = 1
        with VirtualRuntime(cache_size=8, backend=backend, trace_file=str(self.path)) as runtime:
            retried = runtime.execute(call)
            cached = runtime.execute(call)
        self.assertEqual((retried.result, retried.raw_text), (5, final_text))
        self.assertEqual((retried.retries, retried.model_calls, retried.input_tokens, retried.output_tokens), (1, 2, 4, 6))
        self.assertEqual(len(retried.prompts), 1)
        self.assertEqual(len(retried.reasoning_requests), 2)
        self.assertEqual((retried.type_warnings, retried.constraint_warnings), ([], []))
        self.assertIsNone(retried.parse_error)
        self.assertTrue(cached.cached)
        self.assertEqual((cached.result, cached.raw_text, cached.prompts), (retried.result, final_text, retried.prompts))
        self.assertEqual((cached.retries, cached.model_calls, cached.total_tokens), (0, 0, 0))
        self.assertEqual(cached.reasoning_requests, [])
        self.assertEqual(len(backend.calls), 2)
        first, second = self.entries()
        self.assertEqual((first["model_calls"], second["model_calls"]), (2, 0))

    def test_execution_errors_are_traced_without_prediction_parsing_errors(self):
        backend = TraceBackend("def f(x): return x / 0")
        with VirtualRuntime(backend=backend, strategy="code_execute", trace_file=str(self.path)) as runtime:
            invocation = runtime.execute(request())
        entry, = self.entries()
        self.assertEqual(entry["validation_status"], "execution_failed")
        self.assertIn("ZeroDivisionError", entry["execution_error"])
        self.assertEqual(entry["execution_error"], invocation.execution_error)
        self.assertIsNone(entry["parse_error"])
        self.assertIsNone(entry["parse_status"])

    def test_none_return_and_unknown_latency_remain_explicit(self):
        tracer = Tracer(self.path)
        tracer.log(request(), result(result=None, latency_ms=None))
        entry, = self.entries()
        self.assertIsNone(entry["result"])
        self.assertIsNone(entry["parsed_result"])
        self.assertEqual(entry["parsed_type"], "NoneType")
        self.assertIsNone(entry["latency_ms"])

    def test_explicit_falsy_candidates_are_not_replaced_by_the_returned_value(self):
        tracer = Tracer(self.path)
        for candidate in (None, 0, False, ""):
            with self.subTest(candidate=candidate):
                tracer.log(request(), result(), parsed_result=candidate)
                entry = self.entries()[-1]
                self.assertEqual(entry["result"], 5)
                self.assertEqual(entry["parsed_result"], candidate)
                self.assertEqual(entry["parsed_type"], type(candidate).__name__)

    def test_logging_does_not_deepcopy_caller_values_or_include_undeclared_attributes(self):
        class Opaque:
            def __deepcopy__(self, memo):
                raise AssertionError("Caller value was copied")

            def __str__(self):
                return "opaque value"

        snapshot = result(result=Opaque())
        snapshot.extra = "UNDECLARED_ATTRIBUTE_SENTINEL"
        tracer = Tracer(self.path)
        tracer.log(request(), snapshot)
        entry, = self.entries()
        self.assertEqual((entry["result"], entry["parsed_result"]), ("opaque value", "opaque value"))
        self.assertEqual(entry["parsed_type"], "Opaque")
        self.assertNotIn("extra", entry)
        self.assertNotIn("UNDECLARED_ATTRIBUTE_SENTINEL", self.path.read_text(encoding="utf-8"))

    def test_runtime_preserves_candidates_withheld_by_strict_validation(self):
        for mode, returned in (("strict", None), ("warn", "wrong"), ("none", "wrong")):
            with self.subTest(mode=mode):
                backend = TraceBackend(frame_answer('{"result": "wrong"}'))
                with VirtualRuntime(backend=backend, validation_mode=mode, trace_file=self.path) as runtime:
                    invocation = runtime.execute(request())
                entry = self.entries()[-1]
                self.assertEqual((entry["result"], entry["parsed_result"]), (returned, "wrong"))
                self.assertEqual(entry["failure_type"], invocation.failure_type)
                self.assertEqual(entry["validation_status"], invocation.validation_status)
                self.assertEqual(entry["raw_text"], invocation.raw_text)

    def test_runtime_records_parse_failure_diagnostics(self):
        backend = TraceBackend("Unframed response")
        with VirtualRuntime(backend=backend, trace_file=self.path) as runtime:
            invocation = runtime.execute(request())
        entry, = self.entries()
        self.assertEqual(entry["failure_type"], "parse_failure")
        self.assertEqual(entry["parse_status"], "failed")
        self.assertEqual(entry["parse_error"], invocation.parse_error)
        self.assertTrue(entry["parse_error"])
        self.assertEqual(entry["model_calls"], 1)

    def test_cached_invocations_record_zero_new_model_work(self):
        predicate = Mock(return_value=True)
        call = request(constraints=[Constraint("positive", predicate)])
        backend = TraceBackend()
        with VirtualRuntime(cache_size=8, backend=backend, validation_mode="none", trace_file=self.path) as runtime:
            runtime.execute(call)
            runtime.execute(call)
        first, cached = self.entries()
        self.assertEqual((first["call_id"], cached["call_id"]), (1, 2))
        self.assertEqual(first["model_calls"], 1)
        self.assertTrue(cached["cached"])
        self.assertEqual((cached["model_calls"], cached["input_tokens"], cached["output_tokens"], cached["total_tokens"]),
                         (0, 0, 0, 0))
        self.assertEqual(cached["reasoning_requests"], [])
        self.assertEqual(cached["reasoning_observations"], [])
        self.assertEqual(cached["transport_diagnostics"], [])
        self.assertEqual(cached["raw_text"], first["raw_text"])
        self.assertEqual(len(backend.calls), 1)
        predicate.assert_not_called()

    def test_multistage_calls_retain_accumulated_usage(self):
        backend = TraceBackend(frame_answer('{"result": 5}'), frame_answer('{"verdict": "accept", "result": 5}'))
        with VirtualRuntime(backend=backend, strategy="contract", trace_file=self.path) as runtime:
            invocation = runtime.execute(request())
        entry, = self.entries()
        self.assertEqual(entry["model_calls"], 2)
        self.assertEqual((entry["input_tokens"], entry["output_tokens"], entry["total_tokens"]), (4, 6, 10))
        self.assertEqual(entry["raw_text"], invocation.raw_text)
        self.assertEqual(entry["prompts"], invocation.prompts)

    def test_write_failure_does_not_block_prediction_and_attempt_numbers_survive_recovery(self):
        backend = TraceBackend()
        with VirtualRuntime(cache_size=8, backend=backend, trace_file=self.directory) as runtime:
            with self.assertLogs("aicomputing", level="WARNING"):
                first = runtime.execute(request())
            self.assertEqual(first.result, 5)
            self.assertEqual(runtime.tracer.call_count, 1)
            runtime.tracer.path = self.path
            cached = runtime.execute(request())
            self.assertEqual(runtime.tracer.call_count, 2)
        entry, = self.entries()
        self.assertEqual(entry["call_id"], 2)
        self.assertTrue(cached.cached)
        self.assertEqual(len(backend.calls), 1)

    def test_serialization_failure_only_warns(self):
        class Unprintable:
            def __str__(self):
                raise RuntimeError("Cannot format caller object")

        tracer = Tracer(self.path)
        with self.assertLogs("aicomputing", level="WARNING"):
            tracer.log(request(), result(result=Unprintable()))
        self.assertEqual(tracer.call_count, 1)
        self.assertFalse(self.path.exists())

    def test_one_tracer_serializes_concurrent_append_operations(self):
        tracer = Tracer(self.path)
        call, snapshot = request(), result()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: tracer.log(call, snapshot), range(20)))
        entries = self.entries()
        self.assertEqual([entry["call_id"] for entry in entries], list(range(1, 21)))
        self.assertTrue(all(entry["result"] == 5 for entry in entries))
        self.assertEqual(tracer.call_count, 20)


if __name__ == "__main__":
    unittest.main()
