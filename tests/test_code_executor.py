"""Behavior checks for the standard generated-code execution worker."""

from __future__ import annotations

import math
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch
import venv

from aicomputing import code_executor
from aicomputing.code_executor import (
    MAX_ERROR_MESSAGE_CHARS,
    execute_code, execute_code_with_retries, extract_code,
)


class ExecutionRetryTests(unittest.TestCase):
    def outcome(self, status, seconds, *, message=None, line=None):
        return {"status": status, "value": 7 if status == "ok" else None,
                "error_type": {"ok": None, "timeout": "TimeoutExpired", "runtime_error": "ZeroDivisionError"}[status],
                "error_message": message, "error_line": line,
                "elapsed_seconds": seconds}

    def test_timeout_then_success_reuses_program_and_input(self):
        source, inputs = "def answer(n): return n", {"n": 7}
        with patch("aicomputing.code_executor.execute_code", side_effect=[
                self.outcome("timeout", 60, message="Worker exceeded the timeout."), self.outcome("ok", 2)]) as worker:
            result = execute_code_with_retries(source, "answer(n)", inputs)
        self.assertEqual(result["value"], 7)
        self.assertEqual(result["elapsed_seconds"], 62)
        self.assertEqual(len(result["execution_attempts"]), 2)
        self.assertEqual(worker.call_count, 2)
        self.assertEqual(worker.call_args_list[0], worker.call_args_list[1])
        self.assertEqual(worker.call_args.kwargs["timeout_seconds"], 120)
        self.assertEqual(result["execution_attempts"][0]["error_message"], "Worker exceeded the timeout.")
        self.assertIsNone(result["execution_attempts"][0]["error_line"])
        self.assertIsNone(result["error_message"])
        self.assertIsNone(result["execution_attempts"][1]["error_message"])

    def test_exhausts_exactly_three_timeout_attempts(self):
        with patch("aicomputing.code_executor.execute_code", return_value=self.outcome("timeout", 60)) as worker:
            result = execute_code_with_retries("def answer(): return 0", "answer()", {})
        self.assertEqual(worker.call_count, 3)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["elapsed_seconds"], 180)

    def test_runtime_errors_are_not_retried(self):
        failure = self.outcome("runtime_error", 1, message="division by zero", line=2)
        with patch("aicomputing.code_executor.execute_code", return_value=failure) as worker:
            result = execute_code_with_retries("def answer(): return 1 / 0", "answer()", {})
        self.assertEqual(worker.call_count, 1)
        self.assertEqual(result["status"], "runtime_error")
        self.assertEqual(result["execution_attempts"][0]["error_message"], "division by zero")
        self.assertEqual(result["execution_attempts"][0]["error_line"], 2)

    def test_invalid_retry_counts(self):
        for count in (-1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                execute_code_with_retries("def answer(): return 0", "answer()", {}, timeout_retries=count)


class ExtractCodeTests(unittest.TestCase):
    def test_accepts_plain_python_and_one_python_fence(self):
        source = "def twice(value):\n    return value * 2"
        self.assertEqual(extract_code(source).strip(), source)
        self.assertEqual(extract_code("```python\n" + source + "\n```").strip(), source)

    def test_raw_responses_are_extracted_without_syntax_validation(self):
        for source in ("def answer(:\n    return 1", "Here is the answer: 42",
                       "Here is the code:\n```python\ndef answer(): return 1\n```"):
            with self.subTest(source=source):
                self.assertEqual(extract_code(source), source)

    def test_rejects_empty_or_malformed_fences_and_multiple_blocks(self):
        samples = (
            "```python\ndef answer(): return 1\n```\nThis solves it.",
            "```python\ndef answer(): return 1\n```\n```python\ndef other(): return 2\n```",
            "```python\ndef answer(): return 1",
            "```javascript\ndef answer(): return 1\n```",
            "```python\n\n```",
            "",
        )
        for sample in samples:
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                extract_code(sample)

    def test_fences_inside_python_strings_are_preserved(self):
        literals = (
            '"""\n```\nvalue\n```\n"""',
            "'''\n```\nvalue\n```\n'''",
            'r"""\n```\nvalue\n```\n"""',
            'b"""\n```\nvalue\n```\n"""',
            'f"""\n```\n{2 + 3}\n```\n"""',
        )
        for literal in literals:
            source = "def answer():\n    return " + literal
            for newline in ("\n", "\r\n"):
                for fence in ("```python", "```py", "```"):
                    with self.subTest(literal=literal, newline=newline, fence=fence):
                        content = source.replace("\n", newline)
                        response = fence + newline + content + newline + "```"
                        self.assertEqual(extract_code(response), content)

    def test_string_contents_do_not_hide_additional_code_blocks(self):
        first = 'def answer():\n    return """\n```\nvalue\n```\n"""'
        for newline in ("\n", "\r\n"):
            for second_fence in ("```python", "```py", "```"):
                with self.subTest(newline=newline, second_fence=second_fence):
                    response = ("```python\n" + first + "\n```\n" + second_fence
                                + "\ndef other(): return 2\n```").replace("\n", newline)
                    with self.assertRaisesRegex(ValueError, "exactly one Python code block"):
                        extract_code(response)


class ExecuteCodeTests(unittest.TestCase):
    def test_normal_eval_accepts_names_calls_and_explicit_globals(self):
        result = self.run_code(
            "def calculate():\n    return eval('sum(values)', {'values': [2, 3, 4]})",
            "calculate() -> int", {})
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["value"], 9)

    def run_code(self, source, signature, inputs, **options):
        result = execute_code(source, signature, inputs, **options)
        self.assertIn("status", result)
        self.assertIn("value", result)
        self.assertIn("error_type", result)
        self.assertIn("error_message", result)
        self.assertIn("error_line", result)
        if result["status"] == "ok":
            self.assertIsNone(result["error_message"])
            self.assertIsNone(result["error_line"])
        self.assertGreaterEqual(result["elapsed_seconds"], 0)
        return result

    def test_typed_values_roundtrip_without_json_coercion(self):
        value = {
            7: (b"\x00\xff", {2, 5}, frozenset({"a", "b"})),
            (1, 2): [None, True, 2**90, -0.25],
        }
        result = self.run_code(
            "def identity(value):\n    return value",
            "identity(value: Any) -> Any",
            {"value": value},
        )
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["value"], value)
        self.assertIs(type(next(iter(result["value"]))), int)
        self.assertIs(type(result["value"][7]), tuple)
        self.assertIs(type(result["value"][7][0]), bytes)
        self.assertIs(type(result["value"][7][1]), set)
        self.assertIs(type(result["value"][7][2]), frozenset)

    def test_nonfinite_float_inputs_roundtrip_in_nested_containers(self):
        value = {"infinities": [float("inf"), float("-inf")], "nan": (float("nan"),)}
        result = self.run_code("def identity(value): return value", "identity(value: Any) -> Any",
                               {"value": value})
        self.assertEqual(result["status"], "ok", result)
        restored = result["value"]
        self.assertEqual(restored["infinities"], [float("inf"), float("-inf")])
        self.assertIs(type(restored["nan"]), tuple)
        self.assertTrue(math.isnan(restored["nan"][0]))

    def test_nonfinite_floats_generated_in_the_worker_can_be_returned(self):
        for literal in ("inf", "-inf", "nan"):
            with self.subTest(literal=literal):
                result = self.run_code(f"def answer(): return float({literal!r})", "answer() -> float", {})
                self.assertEqual(result["status"], "ok", result)
                self.assertIs(type(result["value"]), float)
                if literal == "nan":
                    self.assertTrue(math.isnan(result["value"]))
                else:
                    self.assertEqual(result["value"], float(literal))

    def test_source_validation_happens_only_in_the_worker(self):
        for source, status in (("def answer(): return 7", "ok"),
                               ("def answer(:", "syntax_error"),
                               ("import os\ndef answer(): return 7", "ok")):
            with self.subTest(status=status):
                with patch("aicomputing.code_executor.ast.parse", side_effect=AssertionError("Parent parsed source")):
                    result = self.run_code(extract_code(source), "answer() -> int", {})
                self.assertEqual(result["status"], status, result)

    def test_backticks_inside_python_strings_are_not_code_fences(self):
        source = 'def answer():\n    return "```python\\nvalue\\n```"'
        for response in (source, "```python\n" + source + "\n```"):
            with self.subTest(response=response):
                result = self.run_code(extract_code(response), "answer() -> str", {})
                self.assertEqual(result["status"], "ok", result)
                self.assertEqual(result["value"], "```python\nvalue\n```")

    def test_raw_and_fenced_multiline_strings_can_contain_fence_delimiter_lines(self):
        source = 'def answer():\n    return """\n```\nvalue\n```\n"""'
        for response in (source, "```python\n" + source + "\n```"):
            with self.subTest(response=response):
                result = self.run_code(extract_code(response), "answer() -> str", {})
                self.assertEqual(result["status"], "ok", result)
                self.assertEqual(result["value"], "\n```\nvalue\n```\n")

    def test_incomplete_strings_with_backticks_remain_syntax_failures(self):
        for source in ('def answer():\n    return """\n```\nvalue',
                       'def answer():\n    return f"""\n```\n{value'):
            outcomes = []
            for response in (source, "```python\n" + source + "\n```"):
                with self.subTest(source=source, response=response):
                    extracted = extract_code(response)
                    self.assertEqual(extracted, source)
                    result = self.run_code(extracted, "answer() -> str", {})
                    self.assertEqual(result["status"], "syntax_error", result)
                    outcomes.append({key: result[key] for key in ("error_type", "error_message", "error_line")})
            self.assertEqual(outcomes[0], outcomes[1])

    def test_executes_algorithm_with_math_import(self):
        source = """import math
def primes(limit: int) -> list[int]:
    result = []
    for candidate in range(2, limit + 1):
        if all(candidate % divisor for divisor in range(2, math.isqrt(candidate) + 1)):
            result.append(candidate)
    return result
"""
        result = self.run_code(source, "primes(limit: int) -> list[int]", {"limit": 19})
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["value"], [2, 3, 5, 7, 11, 13, 17, 19])

    def test_positional_only_inputs_follow_declared_order_not_dictionary_order(self):
        result = self.run_code("def pair(left, right, /):\n    return left, right",
                               "pair(x: int, y: int, /) -> tuple[int, int]", {"y": 23, "x": 19})
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["value"], (19, 23))

    def test_mixed_parameter_kinds_keep_keyword_only_arguments_named(self):
        result = self.run_code("def mix(a, /, b, *, c=3):\n    return a, b, c",
                               "mix(a: int, /, b: int, *, c: int = 3) -> tuple[int, int, int]",
                               {"c": 5, "b": 7, "a": 2})
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["value"], (2, 7, 5))

    def test_omitted_positional_defaults_come_from_the_generated_implementation(self):
        source = "def select(first=11, second=22, /, *, scale=1):\n    return first, second * scale"
        signature = "select(first: int = 1, second: int = 2, /, *, scale: int = 1) -> tuple[int, int]"
        for inputs, expected in (({"second": 4, "scale": 2}, (11, 8)),
                                 ({"first": 5}, (5, 22)), ({"scale": 3}, (11, 66))):
            with self.subTest(inputs=inputs):
                result = self.run_code(source, signature, inputs)
                self.assertEqual(result["status"], "ok", result)
                self.assertEqual(result["value"], expected)

    def test_positional_gap_preserves_default_identity_and_does_not_evaluate_annotations(self):
        prefixes = ["from __future__ import annotations\n"]
        if sys.version_info >= (3, 14):
            prefixes.append("")
        for prefix in prefixes:
            with self.subTest(prefix=prefix):
                source = (prefix + "default = []\n"
                          "def select(first: MissingType = default, second: AlsoMissing = 2, /) -> MissingReturn:\n"
                          "    return first is default, second")
                result = self.run_code(source, "select(first: Any = None, second: int = 2, /) -> tuple[bool, int]",
                                       {"second": 3})
                self.assertEqual(result["status"], "ok", result)
                self.assertEqual(result["value"], (True, 3))

    def test_default_inspection_does_not_run_annotation_expressions(self):
        prefixes = ["from __future__ import annotations\n"]
        if sys.version_info >= (3, 14):
            prefixes.append("")
        for prefix in prefixes:
            with self.subTest(prefix=prefix):
                source = (prefix + "evaluations = []\n"
                          "def annotation():\n    evaluations.append('evaluated')\n    return int\n"
                          "def select(first: annotation() = 11, second=2, /):\n"
                          "    return first, second, len(evaluations)")
                result = self.run_code(source, "select(first: int = 11, second: int = 2, /) -> tuple[int, int, int]",
                                       {"second": 3})
                self.assertEqual(result["status"], "ok", result)
                self.assertEqual(result["value"], (11, 3, 0))

    def test_bound_decorated_partial_and_callable_defaults_do_not_resolve_annotations(self):
        prefixes = ["from __future__ import annotations\n"]
        if sys.version_info >= (3, 14):
            prefixes.append("")
        common = ("from functools import partial, wraps\n"
                  "events=[]\ndef annotation():\n    events.append(1)\n    return int\n"
                  "default=[]\n"
                  "def decorate(function):\n"
                  "    @wraps(function)\n    def wrapped(*args, **kwargs): return function(*args, **kwargs)\n"
                  "    return wrapped\n")
        variants = {
            "bound": "class Holder:\n    def select(self, first: annotation()=default, second=2, /):\n"
                     "        return first is default, second, len(events)\nselect=Holder().select",
            "decorated_bound": "class Holder:\n    @decorate\n"
                               "    def select(self, first: annotation()=default, second=2, /):\n"
                               "        return first is default, second, len(events)\nselect=Holder().select",
            "partial": "def base(prefix, first: annotation()=default, second=2, /):\n"
                       "    return first is default, second, len(events)\nselect=partial(base, 'bound')",
            "callable": "class Holder:\n    def __call__(self, first: annotation()=default, second=2, /):\n"
                        "        return first is default, second, len(events)\nselect=Holder()",
            "classmethod": "class Holder:\n    @classmethod\n"
                           "    def select(cls, first: annotation()=default, second=2, /):\n"
                           "        return first is default, second, len(events)\nselect=Holder.select",
            "custom_signature": "from inspect import Signature, Parameter\n"
                                "def base(first=default, second=2, /): return first is default, second, len(events)\n"
                                "def select(*args, **kwargs): return base(*args, **kwargs)\n"
                                "select.__signature__=Signature([Parameter('first', Parameter.POSITIONAL_ONLY, default=default),"
                                "Parameter('second', Parameter.POSITIONAL_ONLY, default=2)])",
        }
        for prefix in prefixes:
            for name, body in variants.items():
                with self.subTest(prefix=prefix, variant=name):
                    result = self.run_code(prefix + common + body,
                        "select(first: Any=None, second: int=2, /) -> tuple[bool,int,int]", {"second": 4})
                    self.assertEqual(result["status"], "ok", result)
                    self.assertEqual(result["value"], (True, 4, 0))

    def test_missing_annotations_on_bound_methods_and_partials_are_not_resolved(self):
        prefix = "from __future__ import annotations\n" if sys.version_info < (3, 14) else ""
        sources = (
            "class Holder:\n    def select(self, first: MissingType=11, second=22, /) -> MissingReturn:\n"
            "        return first, second\nselect=Holder().select",
            "from functools import partial\ndef base(first: MissingType=11, second=22, /) -> MissingReturn:\n"
            "    return first, second\nselect=partial(base)",
        )
        for source in sources:
            with self.subTest(source=source):
                result = self.run_code(prefix + source, "select(first: int=11, second: int=22, /) -> tuple[int,int]",
                                       {"second": 4})
                self.assertEqual(result["status"], "ok", result)
                self.assertEqual(result["value"], (11, 4))

    def test_malformed_unicode_source_is_a_syntax_failure(self):
        for codepoint in (0xD800, 0xDFFF):
            with self.subTest(codepoint=codepoint):
                source = 'def answer():\n    return "' + chr(codepoint) + '"'
                result = execute_code_with_retries(source, "answer() -> str", {}, timeout_retries=2)
                self.assertEqual(result["status"], "syntax_error", result)
                self.assertEqual(result["error_type"], "UnicodeEncodeError")
                self.assertEqual(len(result["execution_attempts"]), 1)

    def test_unicode_error_raised_by_valid_source_is_a_runtime_failure(self):
        result = self.run_code("def answer(): return chr(233).encode('ascii')", "answer() -> bytes", {})
        self.assertEqual(result["status"], "runtime_error", result)
        self.assertEqual(result["error_type"], "UnicodeEncodeError")

    def test_an_implementation_missing_a_positional_default_remains_an_execution_failure(self):
        result = self.run_code("def select(first, second, /):\n    return first + second",
                               "select(first: int = 1, second: int = 2, /) -> int", {"second": 3})
        self.assertEqual(result["status"], "runtime_error", result)
        self.assertEqual(result["error_type"], "TypeError")
        self.assertIn("first", result["error_message"])

    def test_explicit_none_is_not_treated_as_an_omitted_positional_input(self):
        result = self.run_code("def select(first=11, second=22, /):\n    return first, second",
                               "select(first: Any = 1, second: Any = 2, /) -> tuple[Any, Any]",
                               {"second": None, "first": None})
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["value"], (None, None))

    def test_supports_future_annotations_helpers_and_recursion(self):
        source = """from __future__ import annotations
from typing import List
def factorial(value: int) -> int:
    return 1 if value < 2 else value * factorial(value - 1)
def factorials(values: List[int]) -> List[int]:
    return [factorial(value) for value in values]
"""
        result = self.run_code(
            source, "factorials(values: list[int]) -> list[int]", {"values": [0, 3, 6]}
        )
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["value"], [1, 6, 720])

    def test_normal_filesystem_and_working_directory_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "worker-output.txt"
            result = self.run_code(
                "from pathlib import Path\ndef write_file(path):\n"
                "    Path(path).write_text('written by worker')\n"
                "    return str(Path.cwd())",
                "write_file(path: str) -> str", {"path": str(target)})
            self.assertEqual(result["status"], "ok", result)
            self.assertEqual(Path(result["value"]), Path.cwd())
            self.assertEqual(target.read_text(), "written by worker")

    def test_environment_variables_are_inherited(self):
        source = "import os\ndef setting(): return os.environ.get('AICOMP_WORKER_TEST_SETTING')"
        with patch.dict(os.environ, {"AICOMP_WORKER_TEST_SETTING": "fixture-value"}):
            result = self.run_code(source, "setting() -> str", {})
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["value"], "fixture-value")

    def test_helper_classes_dataclasses_dynamic_imports_and_introspection(self):
        source = """from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Box:
    value: int

def answer(value):
    module = __import__('json')
    box = Box(module.loads(str(value)))
    return type(box).__name__, getattr(box, 'value')
"""
        result = self.run_code(source, "answer(value: int) -> tuple[str, int]", {"value": 42})
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["value"], ("Box", 42))

    def test_working_directory_modules_can_be_imported(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "aicomp_worker_fixture.py").write_text("VALUE = 42\n")
            original_directory = Path.cwd()
            try:
                os.chdir(directory)
                result = self.run_code(
                    "import aicomp_worker_fixture\ndef answer(): return aicomp_worker_fixture.VALUE",
                    "answer() -> int", {})
            finally:
                os.chdir(original_directory)
            self.assertEqual(result["status"], "ok", result)
            self.assertEqual(result["value"], 42)

    def test_subprocesses_are_allowed_and_cleaned_up_after_the_worker_exits(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ready, late = directory / "ready.txt", directory / "late.txt"
            child_source = ("from pathlib import Path; import time; "
                            f"Path({str(ready)!r}).write_text('ready'); "
                            "time.sleep(2); "
                            f"Path({str(late)!r}).write_text('child survived')")
            source = ("import subprocess, sys, time\nfrom pathlib import Path\n"
                      "def start_child():\n"
                      f"    subprocess.Popen([sys.executable, '-c', {child_source!r}])\n"
                      f"    while not Path({str(ready)!r}).exists(): time.sleep(0.01)\n"
                      "    return True")
            result = self.run_code(source, "start_child() -> bool", {}, timeout_seconds=10)
            self.assertEqual(result["status"], "ok", result)
            self.assertIs(result["value"], True)
            self.assertTrue(ready.exists())
            time.sleep(2.2)
            self.assertFalse(late.exists(), "A child process survived worker cleanup")

    def test_reports_syntax_and_runtime_failures(self):
        cases = (
            ("def answer(:\n    return 1", "syntax_error", "SyntaxError"),
            ("def answer():\n    return 1 / 0", "runtime_error", "ZeroDivisionError"),
        )
        for source, status, error_type in cases:
            with self.subTest(status=status):
                result = self.run_code(source, "answer() -> int", {})
                self.assertEqual(result["status"], status, result)
                self.assertEqual(result["error_type"], error_type)
                self.assertIsNone(result["value"])
                self.assertTrue(result["error_message"])
                self.assertEqual(result["error_line"], 1 if status == "syntax_error" else 2)

    def test_runtime_error_identifies_the_innermost_generated_line(self):
        source = "def helper():\n    return missing_name\n\ndef answer():\n    return helper()"
        result = self.run_code(source, "answer() -> int", {})
        self.assertEqual(result["error_type"], "NameError")
        self.assertIn("missing_name", result["error_message"])
        self.assertEqual(result["error_line"], 2)

    def test_diagnostic_messages_are_short_and_single_line(self):
        result = self.run_code("def answer():\n    raise ValueError('first line\\n' + 'x' * 2000)",
                               "answer() -> int", {})
        self.assertEqual(result["error_type"], "ValueError")
        self.assertTrue(result["error_message"].startswith("first line "))
        self.assertLessEqual(len(result["error_message"]), MAX_ERROR_MESSAGE_CHARS)
        self.assertNotIn("\n", result["error_message"])
        self.assertEqual(result["error_line"], 2)

    def test_unprintable_exception_message_does_not_change_the_execution_failure(self):
        result = self.run_code("def answer():\n    raise ValueError(10 ** 5000)", "answer() -> int", {})
        self.assertEqual(result["status"], "runtime_error")
        self.assertEqual(result["error_type"], "ValueError")
        self.assertIsInstance(result["error_message"], str)
        self.assertLessEqual(len(result["error_message"]), MAX_ERROR_MESSAGE_CHARS)
        self.assertEqual(result["error_line"], 2)

    def test_missing_requested_function_is_an_execution_failure(self):
        result = self.run_code("def another():\n    return 1", "answer() -> int", {})
        self.assertEqual(result["status"], "runtime_error", result)
        self.assertIsNone(result["value"])

    def test_infinite_loop_is_terminated(self):
        started = time.monotonic()
        result = self.run_code(
            "def answer():\n    while True:\n        pass",
            "answer() -> int",
            {},
            timeout_seconds=0.3,
        )
        self.assertEqual(result["status"], "timeout", result)
        self.assertIsNone(result["value"])
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertIn("0.3-second timeout", result["error_message"])
        self.assertIsNone(result["error_line"])

    def test_worker_is_terminated_if_cleanup_group_setup_fails(self):
        process = Mock()
        process.poll.return_value = None
        process.pid = 123456
        with patch("aicomputing.code_executor.subprocess.Popen", return_value=process):
            with patch("aicomputing.code_executor._wait_for_worker_ready", return_value=process.pid), \
                    patch("aicomputing.code_executor._windows_job", side_effect=OSError("Group setup failed")), \
                    patch("aicomputing.code_executor.os.killpg", create=True):
                result = self.run_code("def answer(): return 7", "answer() -> int", {})
        self.assertEqual(result["status"], "harness_error")
        self.assertEqual(result["error_type"], "OSError")
        self.assertEqual(result["error_message"], "Group setup failed")
        self.assertIsNone(result["error_line"])
        process.kill.assert_called_once()
        process.wait.assert_called_once()
        process.stdin.write.assert_not_called()
        process.stdin.close.assert_called_once()

    def test_worker_start_signal_is_sent_only_after_cleanup_setup(self):
        process = Mock(pid=123456)
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", 0.1), 0]
        configured = []

        def configure(worker, worker_pid=None):
            self.assertIs(worker, process)
            process.stdin.write.assert_not_called()
            configured.append(True)

        def start(signal):
            self.assertEqual(configured, [True])
            self.assertEqual(signal, code_executor._WORKER_START_SIGNAL)
            return len(signal)

        process.stdin.write.side_effect = start
        with patch("aicomputing.code_executor.subprocess.Popen", return_value=process) as launch, \
                patch("aicomputing.code_executor._wait_for_worker_ready", return_value=process.pid), \
                patch("aicomputing.code_executor._windows_job", side_effect=configure), \
                patch("aicomputing.code_executor.os.killpg", create=True):
            result = self.run_code("def answer(): return 1", "answer() -> int", {}, timeout_seconds=0.1)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(launch.call_args.kwargs["stdin"], subprocess.PIPE)
        process.stdin.write.assert_called_once_with(code_executor._WORKER_START_SIGNAL)

    def test_parent_disconnect_before_start_does_not_execute_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            marker = directory / "source-ran.txt"
            request, result = directory / "request.json", directory / "result.json"
            source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\ndef answer(): return 42"
            request.write_text(json.dumps({"code": source, "signature": "answer() -> int", "inputs": ["dict", []]}))
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            completed = subprocess.run([sys.executable, "-c", code_executor._WORKER_BOOTSTRAP,
                code_executor.__file__, "--worker", str(request), str(result)], input=b"",
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=flags, timeout=10)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(marker.exists())
            self.assertFalse(result.exists())

    def test_start_signal_is_not_visible_to_generated_code_stdin(self):
        result = self.run_code("import sys\ndef answer(): return sys.stdin.read()", "answer() -> str", {})
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["value"], "")

    def test_readiness_wait_is_bounded_and_never_releases_source_after_timeout(self):
        process = Mock(pid=123456)
        process.poll.return_value = None
        with patch("aicomputing.code_executor.subprocess.Popen", return_value=process), \
                patch("aicomputing.code_executor._windows_job") as setup, \
                patch("aicomputing.code_executor.os.killpg", create=True):
            result = self.run_code("def answer(): return 42", "answer() -> int", {}, timeout_seconds=0.02)
        self.assertEqual(result["status"], "timeout", result)
        setup.assert_not_called()
        process.stdin.write.assert_not_called()
        process.kill.assert_called_once()
        process.wait.assert_called_once_with(timeout=code_executor._CLEANUP_TIMEOUT_SECONDS)

    def test_expired_setup_budget_does_not_release_generated_source(self):
        process = Mock(pid=123456)

        def slow_setup(worker, worker_pid=None):
            time.sleep(0.03)

        with patch("aicomputing.code_executor.subprocess.Popen", return_value=process), \
                patch("aicomputing.code_executor._wait_for_worker_ready", return_value=process.pid), \
                patch("aicomputing.code_executor._windows_job", side_effect=slow_setup), \
                patch("aicomputing.code_executor.os.killpg", create=True):
            result = self.run_code("def answer(): return 42", "answer() -> int", {}, timeout_seconds=0.01)
        self.assertEqual(result["status"], "timeout", result)
        process.stdin.write.assert_not_called()
        process.kill.assert_called_once()

    @unittest.skipUnless(os.name == "nt", "Windows job-assignment race regression")
    def test_windows_delayed_job_assignment_precedes_source_and_child_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            started, ready, late = (directory / name for name in ("started", "ready", "late"))
            child = ("from pathlib import Path; import time; "
                     f"Path({str(ready)!r}).write_text('ready'); time.sleep(2); "
                     f"Path({str(late)!r}).write_text('survived')")
            source = ("from pathlib import Path\nimport subprocess, sys, time\n"
                      f"Path({str(started)!r}).write_text('started')\n"
                      "def answer():\n"
                      f"    subprocess.Popen([sys.executable, '-c', {child!r}])\n"
                      f"    while not Path({str(ready)!r}).exists(): time.sleep(0.01)\n"
                      "    return True")
            before_assignment = []
            original_job = code_executor._windows_job

            def delayed_job(process, worker_pid=None):
                time.sleep(0.3)
                before_assignment.append(started.exists())
                return original_job(process, worker_pid=worker_pid)

            with patch("aicomputing.code_executor._windows_job", side_effect=delayed_job):
                result = self.run_code(source, "answer() -> bool", {}, timeout_seconds=10)
            # The probe child has a finite lifetime even if the regression reappears.
            time.sleep(2.2)
            self.assertEqual(result["status"], "ok", result)
            self.assertEqual(before_assignment, [False])
            self.assertTrue(started.exists())
            self.assertTrue(ready.exists())
            self.assertFalse(late.exists(), "A child escaped the worker cleanup job")

    def test_worker_uses_the_virtual_environment_and_its_installed_packages(self):
        script = """
import importlib.util,sys
spec=importlib.util.spec_from_file_location('worker_under_test',sys.argv[1])
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
result=worker.execute_code('import worker_fixture_package\\ndef add(x, y): return worker_fixture_package.add(x, y)', 'add(x: int, y: int) -> int',
                           {'x': 2, 'y': 3}, timeout_seconds=5)
assert result['status']=='ok' and result['value']==5, result
"""
        with tempfile.TemporaryDirectory(prefix="aicomp-venv-test-") as temporary:
            environment = Path(temporary) / "venv"
            venv.EnvBuilder(with_pip=False).create(environment)
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            site = environment / ("Lib/site-packages" if os.name == "nt" else
                                  f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
            site.mkdir(parents=True, exist_ok=True)
            (site / "worker_fixture_package.py").write_text("def add(x, y): return x + y\n")
            completed = subprocess.run([str(python), "-c", script, code_executor.__file__],
                                       capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    @unittest.skipUnless(os.name == "nt", "Windows virtual-environment launcher cleanup regression")
    def test_virtual_environment_worker_children_are_cleaned_up_on_timeout(self):
        script = """
import importlib.util, json, sys, time
from pathlib import Path
spec=importlib.util.spec_from_file_location('worker_under_test',sys.argv[1])
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
directory=Path(sys.argv[2])
ready,late=directory/'child-ready',directory/'child-late'
child=("from pathlib import Path; import time; "
       f"Path({str(ready)!r}).write_text('ready'); time.sleep(3); "
       f"Path({str(late)!r}).write_text('survived')")
source=("import subprocess,sys,time\\ndef answer():\\n"
        f"    subprocess.Popen([sys.executable,'-c',{child!r}])\\n"
        "    while True: time.sleep(0.01)")
result=worker.execute_code(source,'answer() -> int',{},timeout_seconds=2)
time.sleep(3.2)
assert result['status']=='timeout',result
assert ready.exists(),'The child never started'
assert not late.exists(),'A venv worker child survived timeout cleanup'
"""
        with tempfile.TemporaryDirectory(prefix="aicomp-venv-cleanup-") as temporary:
            directory = Path(temporary)
            environment = directory / "venv"
            venv.EnvBuilder(with_pip=False).create(environment)
            completed = subprocess.run([str(environment / "Scripts/python.exe"), "-c", script,
                                        code_executor.__file__, str(directory)],
                                       capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_stdout_cannot_corrupt_the_return_channel(self):
        result = self.run_code(
            "def answer(value):\n    print('not the result: 999')\n    return value + 1",
            "answer(value: int) -> int",
            {"value": 41},
        )
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["value"], 42)

    def test_unsupported_return_type_is_serialization_failure(self):
        result = self.run_code(
            "def answer():\n    return 1 + 2j", "answer() -> Any", {}
        )
        self.assertEqual(result["status"], "serialization_error", result)
        self.assertIsNone(result["value"])
        self.assertEqual(result["error_message"], "Unsupported value type: builtins.complex")
        self.assertIsNone(result["error_line"])

    def test_invalid_timeouts_are_rejected_before_execution(self):
        for timeout in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                execute_code("def answer(): return 1", "answer() -> int", {}, timeout_seconds=timeout)


class WorkerReadinessTests(unittest.TestCase):
    def test_transient_file_sharing_errors_are_retried_within_the_startup_budget(self):
        process = Mock()
        process.poll.return_value = None
        with patch.object(Path, "read_text", side_effect=[PermissionError("temporarily shared"),
                FileNotFoundError(), '{"status": "ready", "pid": 1234}']) as read, \
                patch.object(code_executor.time, "monotonic", return_value=1), \
                patch.object(code_executor.time, "sleep") as sleep:
            pid = code_executor._wait_for_worker_ready(process, Path("result.json"), 2, 1)
        self.assertEqual(pid, 1234)
        self.assertEqual(read.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_process_exit_before_readiness_reports_startup_failure(self):
        process = Mock(returncode=9)
        process.poll.return_value = 9
        with patch.object(Path, "read_text", side_effect=FileNotFoundError()), \
                self.assertRaisesRegex(OSError, r"exited before startup completed \(code 9\)"):
            code_executor._wait_for_worker_ready(process, Path("result.json"), 2, 1)

    def test_persistent_access_error_is_a_harness_failure_at_the_deadline(self):
        process = Mock()
        process.poll.return_value = None
        with patch.object(Path, "read_text", side_effect=PermissionError("not readable")), \
                patch.object(code_executor.time, "monotonic", return_value=2), \
                patch.object(code_executor.time, "sleep") as sleep, \
                self.assertRaisesRegex(PermissionError, "not readable"):
            code_executor._wait_for_worker_ready(process, Path("result.json"), 2, 1)
        sleep.assert_not_called()

    def test_invalid_startup_metadata_is_rejected(self):
        for value in (None, [], {}, {"status": "ok", "pid": 1}, {"status": "ready", "pid": True},
                      {"status": "ready", "pid": 0}, {"status": "ready", "pid": "1234"}):
            with self.subTest(value=value), patch.object(Path, "read_text", return_value=json.dumps(value)), \
                    self.assertRaisesRegex(OSError, "invalid startup response"):
                code_executor._wait_for_worker_ready(Mock(), Path("result.json"), 2, 1)


class WorkerCleanupTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.failures = {}
        self.process = Mock(pid=123456, returncode=0)
        self.process.stdin = SimpleNamespace(write=Mock(return_value=1),
                                             close=Mock(side_effect=lambda: self.action("stdin")))
        self.process.kill.side_effect = lambda: self.action("kill")
        self.process.wait.side_effect = lambda **options: self.action("wait", 0)
        self.close_job = Mock(side_effect=lambda: self.action("job"))
        self.temporary = SimpleNamespace(cleanup=Mock(side_effect=lambda: self.action("temporary")))
        self.group_kill = Mock(side_effect=lambda *args: self.action("group"))
        for patcher in (
            patch.object(code_executor, "os", SimpleNamespace(name="posix", killpg=self.group_kill)),
            patch.object(code_executor, "signal", SimpleNamespace(SIGKILL=9)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def action(self, name, value=None):
        self.events.append(name)
        if name in self.failures:
            raise self.failures[name]
        return value

    def launch(self, command, **options):
        self.worker_directory = Path(command[-1]).parent
        Path(command[-1]).write_text(json.dumps({"status": "ok", "value": ["int", "0x2a"],
            "error_type": None, "error_message": None, "error_line": None}))
        return self.process

    def run_execution(self):
        with patch("aicomputing.code_executor.subprocess.Popen", side_effect=self.launch), \
                patch("aicomputing.code_executor._wait_for_worker_ready", return_value=self.process.pid), \
                patch("aicomputing.code_executor._windows_job", return_value=self.close_job):
            return execute_code("def answer(): return 42", "answer() -> int", {}, timeout_seconds=0.1)

    def test_each_cleanup_failure_still_attempts_every_step(self):
        for step in ("stdin", "job", "group", "kill", "wait", "temporary"):
            with self.subTest(step=step):
                self.events.clear()
                self.process.wait.reset_mock()
                failure = OSError(step + " failed")
                self.failures = {step: failure}
                with self.assertRaises(OSError) as caught:
                    code_executor._cleanup_execution(self.process, self.close_job, self.temporary)
                self.assertIs(caught.exception, failure)
                self.assertEqual(self.events, ["stdin", "job", "group", "kill", "wait", "temporary"])
                timeout = self.process.wait.call_args.kwargs["timeout"]
                self.assertTrue(math.isfinite(timeout) and timeout > 0)

    def test_multiple_cleanup_failures_keep_the_first_error_and_report_additional_types(self):
        first = OSError("private job detail")
        self.failures = {"job": first, "kill": RuntimeError("private kill detail"),
                         "temporary": ValueError("private directory detail")}
        with self.assertLogs("aicomputing", level="WARNING") as logs:
            with self.assertRaises(OSError) as caught:
                code_executor._cleanup_execution(self.process, self.close_job, self.temporary)
        self.assertIs(caught.exception, first)
        self.assertEqual(self.events, ["stdin", "job", "group", "kill", "wait", "temporary"])
        self.assertIn("RuntimeError", logs.output[0])
        self.assertIn("ValueError", logs.output[0])
        self.assertNotIn("private", logs.output[0])

    def test_an_already_missing_process_group_is_not_a_cleanup_failure(self):
        self.failures = {"group": ProcessLookupError()}
        code_executor._cleanup_execution(self.process, self.close_job, self.temporary)
        self.assertEqual(self.events, ["stdin", "job", "group", "kill", "wait", "temporary"])

    def test_cancellation_during_cleanup_is_not_hidden_by_an_earlier_cleanup_error(self):
        cancellation = KeyboardInterrupt("cancel cleanup")
        self.failures = {"job": OSError("job failed"), "kill": cancellation}
        with self.assertLogs("aicomputing", level="WARNING"):
            with self.assertRaises(KeyboardInterrupt) as caught:
                code_executor._cleanup_execution(self.process, self.close_job, self.temporary)
        self.assertIs(caught.exception, cancellation)
        self.assertEqual(self.events, ["stdin", "job", "group", "kill", "wait", "temporary"])

    def test_timeout_cleanup_failure_still_terminates_and_reaps_the_worker(self):
        self.process.wait.side_effect = [subprocess.TimeoutExpired("worker", 0.1), 0]
        self.failures = {"job": OSError("job cleanup failed")}
        result = self.run_execution()
        self.assertEqual(result["status"], "harness_error")
        self.assertEqual(result["error_message"], "job cleanup failed")
        self.process.kill.assert_called_once()
        self.assertGreater(self.process.wait.call_args_list[0].kwargs["timeout"], 0)
        self.assertLessEqual(self.process.wait.call_args_list[0].kwargs["timeout"], 0.1)
        self.assertEqual(self.process.wait.call_args_list[1], call(timeout=code_executor._CLEANUP_TIMEOUT_SECONDS))
        self.assertFalse(self.worker_directory.exists())

    def test_worker_wait_respects_timeout_despite_clock_rounding(self):
        # This clock value makes (clock + 0.1) - clock slightly exceed 0.1.
        with patch.object(code_executor.time, "monotonic", return_value=100000.0):
            result = self.run_execution()
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["value"], 42)
        self.assertEqual(self.process.wait.call_args_list[0], call(timeout=0.1))

    def test_active_execution_errors_and_cancellation_survive_cleanup_failures(self):
        for original in (OSError("original worker error"), KeyboardInterrupt("cancelled"), SystemExit(7)):
            with self.subTest(original=type(original).__name__):
                self.process.kill.reset_mock()
                self.process.wait.reset_mock()
                self.process.wait.side_effect = [original, 0]
                self.failures = {"job": OSError("secondary cleanup error")}
                with self.assertLogs("aicomputing", level="WARNING") as logs:
                    if isinstance(original, Exception):
                        result = self.run_execution()
                        self.assertEqual(result["error_message"], "original worker error")
                        self.assertEqual(result["status"], "harness_error")
                    else:
                        with self.assertRaises(type(original)) as caught:
                            self.run_execution()
                        self.assertIs(caught.exception, original)
                self.assertIn("preserving the original error", logs.output[0])
                self.process.kill.assert_called_once()
                self.assertEqual(self.process.wait.call_count, 2)
                self.assertFalse(self.worker_directory.exists())

    def test_a_callers_exception_handler_does_not_hide_a_cleanup_only_failure(self):
        self.failures = {"job": OSError("job cleanup failed")}
        try:
            raise ValueError("unrelated caller error")
        except ValueError:
            result = self.run_execution()
        self.assertEqual(result["status"], "harness_error")
        self.assertEqual(result["error_message"], "job cleanup failed")
        self.process.kill.assert_called_once()
        self.assertEqual(self.process.wait.call_count, 2)
        self.assertFalse(self.worker_directory.exists())

    def test_temporary_directory_cleanup_failure_does_not_replace_cancellation(self):
        cancellation = KeyboardInterrupt("cancelled")
        self.process.wait.side_effect = [cancellation, 0]
        self.failures = {"job": OSError("job cleanup failed")}
        original_factory = tempfile.TemporaryDirectory

        def make_temporary(**options):
            temporary = original_factory(**options)
            original_cleanup = temporary.cleanup
            self.addCleanup(original_cleanup)

            def failing_cleanup():
                original_cleanup()
                raise OSError("directory cleanup failed")

            temporary.cleanup = failing_cleanup
            return temporary

        with patch("aicomputing.code_executor.tempfile.TemporaryDirectory", side_effect=make_temporary):
            with self.assertLogs("aicomputing", level="WARNING"):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    self.run_execution()
        self.assertIs(caught.exception, cancellation)
        self.process.kill.assert_called_once()
        self.assertEqual(self.process.wait.call_count, 2)
        self.assertFalse(self.worker_directory.exists())


class WindowsJobTests(unittest.TestCase):
    def setUp(self):
        self.last_error = 0
        self.process = Mock(pid=123456, _handle=456)
        self.kernel = SimpleNamespace(
            CreateJobObjectW=Mock(return_value=123),
            SetInformationJobObject=Mock(return_value=1),
            AssignProcessToJobObject=Mock(return_value=1),
            OpenProcess=Mock(return_value=789),
            CloseHandle=Mock(side_effect=self.close_handle),
        )
        self.loader = Mock(return_value=self.kernel)
        self.error_factory = Mock(side_effect=self.make_win_error)
        for patcher in (
            patch.object(code_executor, "os", SimpleNamespace(name="nt")),
            patch("ctypes.WinDLL", self.loader, create=True),
            patch("ctypes.get_last_error", lambda: self.last_error, create=True),
            patch("ctypes.FormatError", lambda code: f"Native failure {code}\r\n", create=True),
            patch("ctypes.WinError", self.error_factory, create=True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def close_handle(self, handle):
        # Native cleanup may overwrite the error recorded by the failed call.
        self.last_error = 0
        return 1

    @staticmethod
    def make_win_error(code, message):
        error = OSError(code, message)
        error.winerror = code
        return error

    def test_setup_failures_preserve_windows_errors_and_close_owned_handles(self):
        operations = ("CreateJobObjectW", "SetInformationJobObject", "AssignProcessToJobObject")
        for operation, error_code in zip(operations, (5, 87, 6)):
            with self.subTest(operation=operation):
                for name in operations:
                    api = getattr(self.kernel, name)
                    api.reset_mock()
                    api.return_value = 123 if name == "CreateJobObjectW" else 1
                self.kernel.CloseHandle.reset_mock()
                self.error_factory.reset_mock()
                getattr(self.kernel, operation).return_value = 0
                self.last_error = error_code

                with self.assertRaises(OSError) as caught:
                    code_executor._windows_job(self.process)

                self.assertEqual(caught.exception.winerror, error_code)
                self.assertIn(operation, str(caught.exception))
                self.error_factory.assert_called_once_with(error_code, f"{operation} failed: Native failure {error_code}")
                if operation == "CreateJobObjectW":
                    self.kernel.CloseHandle.assert_not_called()
                    self.kernel.SetInformationJobObject.assert_not_called()
                else:
                    self.kernel.CloseHandle.assert_called_once_with(123)
                    self.assertEqual(self.last_error, 0)
                if operation != "AssignProcessToJobObject":
                    self.kernel.AssignProcessToJobObject.assert_not_called()

    def test_success_returns_a_deferred_cleanup_callback(self):
        close_job = code_executor._windows_job(self.process)
        self.assertTrue(callable(close_job))
        self.error_factory.assert_not_called()
        self.kernel.CloseHandle.assert_not_called()
        close_job()
        self.kernel.CloseHandle.assert_called_once_with(123)

    def test_close_handle_failure_reports_the_windows_error(self):
        close_job = code_executor._windows_job(self.process)
        self.kernel.CloseHandle.side_effect = None
        self.kernel.CloseHandle.return_value = 0
        self.last_error = 6
        with self.assertRaises(OSError) as caught:
            close_job()
        self.assertEqual(caught.exception.winerror, 6)
        self.assertIn("CloseHandle", str(caught.exception))

    def test_virtual_environment_worker_pid_is_assigned_instead_of_its_launcher(self):
        close_job = code_executor._windows_job(self.process, worker_pid=777)
        self.kernel.OpenProcess.assert_called_once_with(0x0100 | 0x0001, False, 777)
        self.kernel.AssignProcessToJobObject.assert_called_once_with(123, 789)
        self.kernel.CloseHandle.assert_called_once_with(789)
        close_job()
        self.assertEqual(self.kernel.CloseHandle.call_args_list, [call(789), call(123)])

    def test_worker_process_handle_and_job_are_closed_on_setup_failure(self):
        for operation in ("OpenProcess", "AssignProcessToJobObject"):
            with self.subTest(operation=operation):
                self.kernel.OpenProcess.return_value = 0 if operation == "OpenProcess" else 789
                self.kernel.AssignProcessToJobObject.return_value = 0 if operation == "AssignProcessToJobObject" else 1
                self.kernel.CloseHandle.reset_mock()
                self.last_error = 5
                with self.assertRaises(OSError) as caught:
                    code_executor._windows_job(self.process, worker_pid=777)
                self.assertEqual(caught.exception.winerror, 5)
                self.assertIn(operation, str(caught.exception))
                expected = [call(123)] if operation == "OpenProcess" else [call(789), call(123)]
                self.assertEqual(self.kernel.CloseHandle.call_args_list, expected)

    def test_other_platforms_skip_windows_api_calls(self):
        with patch.object(code_executor, "os", SimpleNamespace(name="posix")):
            self.assertIsNone(code_executor._windows_job(self.process))
        self.loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
