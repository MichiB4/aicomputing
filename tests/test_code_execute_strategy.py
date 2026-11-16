"""Public API coverage for generated-code execution, without paid inference."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from aicomputing import CodeExecutionError, ExecuteRequest, VirtualRuntime, create_strategy
from aicomputing.backends.base import Backend, GenerateResponse
from aicomputing.request import FailureType
from aicomputing.strategies import list_strategies


class SourceBackend(Backend):
    name = "source_fixture"
    default_model = "offline-model"

    def __init__(self, source, *, finish_reason="stop"):
        self.source, self.finish_reason = source, finish_reason
        self.calls = []

    def cache_state(self):
        return {"source": self.source, "finish_reason": self.finish_reason}

    def generate(self, prompt, **options):
        return self.generate_chat([{"role": "user", "content": prompt}], **options)

    def generate_chat(self, messages, **options):
        self.calls.append((messages, options))
        return GenerateResponse(
            text=self.source, model=options.get("model") or self.default_model,
            input_tokens=11, output_tokens=17,
            finish_reason=self.finish_reason, provider_finish_reason=self.finish_reason,
            requested_reasoning={"reasoning_effort": "none"},
            observed_reasoning={"reasoning_tokens": 0},
        )


def request(**options):
    return ExecuteRequest(signature="add(x: int, y: int) -> int", docstring="Return x + y.",
                          inputs={"x": 19, "y": 23}, **options)


class CodeExecuteStrategyTests(unittest.TestCase):
    def test_registered_runtime_strategy_returns_value_source_usage_and_trace(self):
        self.assertIn("code_execute", list_strategies())
        backend = SourceBackend("def add(x, y):\n    return x + y")
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "trace.jsonl"
            with VirtualRuntime(backend=backend, strategy="code_execute", trace_file=str(trace)) as runtime:
                result = runtime.execute(request())
            evidence = json.loads(trace.read_text(encoding="utf-8"))
        self.assertEqual(result.result, 42)
        self.assertEqual(result.validation_status, "valid")
        self.assertEqual(result.generated_code, backend.source)
        self.assertEqual(result.execution_details["status"], "ok")
        self.assertEqual(result.answer_format, "python_code_v1")
        self.assertEqual((result.model_calls, result.input_tokens, result.output_tokens, result.total_tokens), (1, 11, 17, 28))
        self.assertEqual(result.reasoning_requests, [{"reasoning_effort": "none"}])
        self.assertEqual(result.prompts, [backend.calls[0][0][0]["content"]])
        self.assertEqual(evidence["generated_code"], backend.source)
        self.assertEqual(evidence["execution_details"]["status"], "ok")
        self.assertIsNone(evidence["execution_details"]["error_message"])
        self.assertIsNone(evidence["execution_details"]["error_line"])
        self.assertEqual(evidence["validation_status"], "valid")
        self.assertEqual((evidence["input_tokens"], evidence["output_tokens"], evidence["total_tokens"]), (11, 17, 28))

    def test_custom_word_problem_mode_and_checkpoint_callback(self):
        backend = SourceBackend("def solve(question):\n    return 20 / 2")
        seen = []
        strategy = create_strategy("code_execute", task_mode="word_problem", on_generated_code=seen.append)
        with VirtualRuntime(backend=backend, strategy=strategy) as runtime:
            result = runtime.execute(ExecuteRequest(signature="solve(question: str) -> float",
                docstring="Solve the supplied problem.", inputs={"question": "What is half of twenty?"}))
        self.assertEqual(result.result, 10.0)
        self.assertEqual(seen, [backend.source])
        self.assertIn("Write arithmetic code for this specific supplied word problem", result.prompts[0])

    def test_execution_failures_never_regenerate_or_get_cached(self):
        for source, status in [
            ("Here is the answer: 42", "syntax_error"),
            ("", "format_error"),
            ("```javascript\ndef add(x, y): return x + y\n```", "format_error"),
            ("```python\ndef add(x, y)\n    return x + y\n```", "syntax_error"),
            ("def add(x, y): return x / 0", "runtime_error"),
        ]:
            with self.subTest(status=status):
                backend = SourceBackend(source)
                with VirtualRuntime(cache_size=8, backend=backend, strategy="code_execute") as runtime:
                    for _ in range(2):
                        result = runtime.execute(request(max_retries=4))
                        self.assertEqual(result.execution_details["status"], status)
                        self.assertEqual(result.validation_status, "execution_failed")
                        self.assertEqual(result.failure_type, FailureType.EXECUTION)
                        self.assertIsNone(result.result)
                        self.assertEqual(result.retries, 0)
                        self.assertFalse(result.cached)
                        self.assertTrue(result.execution_details["error_message"])
                        self.assertIn(result.execution_details["error_message"], result.execution_error)
                        self.assertIsNone(result.parse_error)
                        self.assertIsNone(result.parse_status)
                self.assertEqual(len(backend.calls), 2)

    def test_raw_and_fenced_broken_python_have_identical_syntax_diagnostics(self):
        source = "def add(x, y)\n    return x + y"
        diagnostics = []
        for response in (source, "```python\n" + source + "\n```"):
            with self.subTest(response=response):
                backend = SourceBackend(response)
                with VirtualRuntime(backend=backend, strategy="code_execute") as runtime:
                    result = runtime.execute(request())
                execution = result.execution_details
                self.assertEqual(execution["status"], "syntax_error")
                self.assertEqual(result.generated_code, source)
                self.assertEqual(result.raw_text, response)
                self.assertEqual(result.model_calls, 1)
                self.assertEqual(len(execution["execution_attempts"]), 1)
                diagnostics.append({key: execution[key] for key in ("status", "error_type", "error_message", "error_line")})
        self.assertEqual(diagnostics[0], diagnostics[1])

    def test_local_timeout_retry_reuses_source_without_another_model_call(self):
        backend = SourceBackend("def add(x, y): return x + y")
        strategy = create_strategy("code_execute", execution_timeout=180, execution_retries=1)
        outcomes = [dict(status="timeout", value=None, error_type="TimeoutExpired", elapsed_seconds=180,
                         error_message="Worker exceeded the timeout.", error_line=None),
                    dict(status="ok", value=42, error_type=None, elapsed_seconds=1,
                         error_message=None, error_line=None)]
        with patch("aicomputing.code_executor.execute_code", side_effect=outcomes) as worker:
            with VirtualRuntime(backend=backend, strategy=strategy) as runtime:
                result = runtime.execute(request())
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(worker.call_count, 2)
        self.assertEqual(worker.call_args_list[0], worker.call_args_list[1])
        self.assertEqual(worker.call_args.kwargs["timeout_seconds"], 180)
        self.assertEqual(result.result, 42)
        self.assertEqual(len(result.execution_details["execution_attempts"]), 2)

    def test_truncated_source_is_not_executed_or_retried_with_a_larger_cap(self):
        backend = SourceBackend("def add(x, y): return x + y", finish_reason="length")
        with patch("aicomputing.strategies.code_execute.execute_code_with_retries") as worker:
            with VirtualRuntime(backend=backend, strategy="code_execute") as runtime:
                result = runtime.execute(request(max_new_tokens=32))
        worker.assert_not_called()
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0][1]["max_new_tokens"], 32)
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(result.validation_status, "execution_failed")
        self.assertEqual(result.failure_type, FailureType.EXECUTION)
        self.assertIn("truncated", result.execution_details["error_message"])
        self.assertIsNone(result.execution_details["error_line"])

    def test_typed_worker_value_is_preserved_when_evaluation_disables_output_conversion(self):
        backend = SourceBackend("def add(x, y): return (x, {y})")
        with VirtualRuntime(backend=backend, strategy="code_execute", validation_mode="none") as runtime:
            result = runtime.execute(request(max_retries=0))
        self.assertEqual(result.result, (19, {23}))
        self.assertIs(type(result.result), tuple)
        self.assertIs(type(result.result[1]), set)
        self.assertEqual(result.validation_status, "not_checked")

    def test_normal_runtime_output_validation_still_applies_without_regeneration(self):
        backend = SourceBackend("def add(x, y): return [x, y]")
        with VirtualRuntime(backend=backend, strategy="code_execute", validation_mode="strict") as runtime:
            result = runtime.execute(request(max_retries=3))
        self.assertIsNone(result.result)
        self.assertEqual(result.validation_status, "invalid_type")
        self.assertEqual(result.failure_type, FailureType.STRUCTURAL_VALIDATION)
        self.assertEqual(result.execution_details["value"], [19, 23])
        self.assertEqual(len(backend.calls), 1)

    def test_exhausted_worker_timeouts_use_the_execution_label_in_every_output_mode(self):
        details = dict(status="timeout", value=None, error_type="TimeoutExpired", elapsed_seconds=1,
                       error_message="Worker exceeded the timeout.", error_line=None)
        for mode in ("strict", "warn", "none"):
            with self.subTest(mode=mode):
                backend = SourceBackend("def add(x, y): return x + y")
                with patch("aicomputing.strategies.code_execute.execute_code_with_retries", return_value=details):
                    with VirtualRuntime(backend=backend, strategy="code_execute", validation_mode=mode) as runtime:
                        result = runtime.execute(request())
                self.assertEqual((result.failure_type, result.validation_status),
                                 ("execution_failure", "execution_failed"))
                self.assertEqual(result.execution_details["status"], "timeout")
                self.assertIsNone(result.result)
                self.assertEqual(len(backend.calls), 1)

    def test_none_return_is_a_successful_execution(self):
        backend = SourceBackend("def clear(): return None")
        with VirtualRuntime(backend=backend, strategy="code_execute") as runtime:
            result = runtime.execute(ExecuteRequest(signature="clear() -> None", docstring="Return None.", inputs={}))
        self.assertEqual(result.validation_status, "valid")
        self.assertIsNone(result.parse_status)
        self.assertIsNone(result.parse_error)
        self.assertIsNone(result.execution_error)
        self.assertIsNone(result.result)

    def test_positional_and_keyword_only_parameters_keep_the_input_dictionary_api(self):
        backend = SourceBackend("def combine(left, /, middle, *, right=3):\n    return left + middle + right")
        inputs = {"middle": 4, "left": 2}
        with VirtualRuntime(backend=backend, strategy="code_execute") as runtime:
            result = runtime.execute(ExecuteRequest(
                signature="combine(left: int, /, middle: int, *, right: int = 3) -> int",
                docstring="Return the sum of the three arguments.", inputs=inputs))
        self.assertEqual(result.result, 9)
        self.assertEqual(result.validation_status, "valid")
        self.assertEqual(result.execution_details["status"], "ok")
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(inputs, {"middle": 4, "left": 2})

    def test_nonfinite_float_inputs_and_outputs_follow_normal_validation_modes(self):
        for mode in ("strict", "warn", "none"):
            for value in (float("inf"), float("-inf"), float("nan")):
                with self.subTest(mode=mode, value=value):
                    backend = SourceBackend("def identity(x): return x")
                    task = ExecuteRequest(signature="identity(x: float) -> float",
                                          docstring="Return the supplied value.", inputs={"x": value})
                    with VirtualRuntime(backend=backend, strategy="code_execute", validation_mode=mode) as runtime:
                        result = runtime.execute(task)
                    self.assertIsNone(result.failure_type)
                    self.assertIsNone(result.execution_error)
                    self.assertEqual(result.execution_details["status"], "ok")
                    self.assertEqual(result.validation_status, "not_checked" if mode == "none" else "valid")
                    self.assertEqual(result.model_calls, 1)
                    if math.isnan(value):
                        self.assertTrue(math.isnan(result.result))
                    else:
                        self.assertEqual(result.result, value)

    def test_worker_exception_without_a_message_remains_an_execution_failure(self):
        backend = SourceBackend("def add(x, y): raise ValueError()")
        with VirtualRuntime(backend=backend, strategy="code_execute") as runtime:
            result = runtime.execute(request())
        self.assertEqual(result.failure_type, "execution_failure")
        self.assertEqual(result.execution_details["error_type"], "ValueError")
        self.assertEqual(result.execution_details["error_message"], "")

    def test_malformed_unicode_reply_is_recorded_without_aborting_or_regenerating(self):
        for fenced in (False, True):
            with self.subTest(fenced=fenced), tempfile.TemporaryDirectory() as temporary:
                source = 'def answer():\n    return """\n```\n' + chr(0xD800) + '\n```\n"""'
                response = "```python\n" + source + "\n```" if fenced else source
                backend = SourceBackend(response)
                trace = Path(temporary) / "trace.jsonl"
                with VirtualRuntime(backend=backend, strategy="code_execute", trace_file=trace) as runtime:
                    result = runtime.execute(ExecuteRequest(signature="answer() -> str", docstring="Return text.",
                                                           inputs={}, max_retries=3))
                self.assertEqual(result.failure_type, "execution_failure")
                self.assertEqual(result.execution_details["status"], "syntax_error")
                self.assertEqual(result.raw_text, response)
                self.assertEqual(result.generated_code, source)
                self.assertEqual(result.model_calls, 1)
                self.assertEqual(len(result.execution_details["execution_attempts"]), 1)
                evidence = json.loads(trace.read_text(encoding="utf-8"))
                self.assertEqual(evidence["raw_text"], response)
                self.assertEqual(evidence["generated_code"], source)

    def test_harness_error_propagates_with_execution_evidence(self):
        backend = SourceBackend("def add(x, y): return x + y")
        details = dict(status="harness_error", value=None, error_type="OSError",
                       error_message="AssignProcessToJobObject failed: access denied", elapsed_seconds=0)
        with patch("aicomputing.strategies.code_execute.execute_code_with_retries", return_value=details):
            with VirtualRuntime(backend=backend, strategy="code_execute") as runtime:
                with self.assertRaises(CodeExecutionError) as caught:
                    runtime.execute(request())
        self.assertEqual(caught.exception.generated_code, backend.source)
        self.assertEqual(caught.exception.execution_details, details)
        self.assertEqual(str(caught.exception),
                         "Python worker failed: OSError: AssignProcessToJobObject failed: access denied")
        self.assertEqual(len(backend.calls), 1)

    def test_request_can_select_code_execution_over_runtime_prediction_default(self):
        backend = SourceBackend("def add(x, y): return x + y")
        with VirtualRuntime(backend=backend, strategy="direct") as runtime:
            result = runtime.execute(request(strategy="code_execute"))
        self.assertEqual(result.result, 42)
        self.assertEqual(result.strategy, "code_execute")

    def test_cache_hit_records_no_new_model_or_worker_call(self):
        backend = SourceBackend("def add(x, y): return x + y")
        with VirtualRuntime(cache_size=8, backend=backend, strategy="code_execute") as runtime:
            first = runtime.execute(request())
            with patch("aicomputing.strategies.code_execute.execute_code_with_retries") as worker:
                second = runtime.execute(request())
        worker.assert_not_called()
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual((second.result, second.model_calls, second.total_tokens), (42, 0, 0))
        self.assertEqual(len(backend.calls), 1)

    def test_invalid_worker_configuration_is_rejected_before_generation(self):
        for options in [dict(execution_timeout=x) for x in (0, -1, True, float("nan"), float("inf"))] + [
            dict(execution_retries=-1), dict(execution_retries=True), dict(task_mode="gsm8k")
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                create_strategy("code_execute", **options)

    def test_invalid_changed_settings_fail_before_generation_in_every_mode(self):
        cases = [("execution_timeout", value, ValueError)
                 for value in (0, -1, True, False, float("nan"), float("inf"), "120", None, [])]
        cases += [("execution_retries", value, ValueError) for value in (-1, True, False, 1.5, "2", None)]
        cases += [("task_mode", value, ValueError) for value in ("gsm8k", "", None)]
        cases += [("on_generated_code", value, TypeError) for value in (False, 0, "callback", [])]
        for mode in ("strict", "warn", "none"):
            for cache_size in (0, 8):
                for attribute, value, error in cases:
                    with self.subTest(mode=mode, cache_size=cache_size, attribute=attribute, value=value):
                        backend = SourceBackend("def add(x, y): return x + y")
                        with patch("aicomputing.strategies.code_execute.execute_code_with_retries") as worker:
                            with VirtualRuntime(backend=backend, strategy="code_execute", cache_size=cache_size,
                                                validation_mode=mode) as runtime:
                                setattr(runtime.strategy, attribute, value)
                                with self.assertRaisesRegex(error, attribute):
                                    runtime.execute(request())
                            worker.assert_not_called()
                        self.assertFalse(backend.calls)

    def test_valid_changed_settings_are_used_for_generation_checkpoint_and_execution(self):
        backend = SourceBackend("def solve(question): return 20 / 2")
        checkpoint = Mock()
        outcome = dict(status="ok", value=10.0, error_type=None, error_message=None, error_line=None,
                       elapsed_seconds=0.0, execution_attempts=[])
        task = ExecuteRequest(signature="solve(question: str) -> float", docstring="Solve the supplied problem.",
                              inputs={"question": "What is half of twenty?"})
        with patch("aicomputing.strategies.code_execute.execute_code_with_retries", return_value=outcome) as worker:
            with VirtualRuntime(backend=backend, strategy="code_execute") as runtime:
                runtime.strategy.execution_timeout = 5.0
                runtime.strategy.execution_retries = 1
                runtime.strategy.task_mode = "word_problem"
                runtime.strategy.on_generated_code = checkpoint
                result = runtime.execute(task)
        self.assertEqual(result.result, 10.0)
        self.assertEqual(len(backend.calls), 1)
        self.assertIn("Write arithmetic code for this specific supplied word problem", result.prompts[0])
        checkpoint.assert_called_once_with(backend.source)
        worker.assert_called_once_with(backend.source, task.signature, task.inputs,
                                       timeout_seconds=5.0, timeout_retries=1)


if __name__ == "__main__":
    unittest.main()
