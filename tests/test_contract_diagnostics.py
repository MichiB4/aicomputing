"""Selected-answer provenance and checker diagnostics remain independently visible."""

import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest

from aicomputing import Backend, ExecuteRequest, ParseStatus, VirtualRuntime
from aicomputing.answer_protocol import frame_answer
from aicomputing.backends.base import GenerateResponse
from evaluation.benchmarks.result_schema import RESULT_COLUMNS, build_row


class ContractBackend(Backend):
    name = "offline"
    default_model = "offline"

    def __init__(self, prediction, verdict):
        self.replies = (prediction, verdict)
        self.calls = []

    def cache_state(self):
        return {"replies": self.replies}

    def generate(self, prompt, **settings):
        payload, finish, original_finish = self.replies[len(self.calls) % len(self.replies)]
        self.calls.append(prompt)
        return GenerateResponse(text=frame_answer(payload), model="offline", finish_reason=finish,
                                provider_finish_reason=original_finish, input_tokens=3, output_tokens=5)


def request(signature="f() -> int"):
    return ExecuteRequest(signature=signature, docstring="Return the required value.", inputs={}, max_retries=0)


class ContractDiagnosticTests(unittest.TestCase):
    def test_answer_selection_and_checker_evidence_survive_public_and_evaluation_results(self):
        defaults = dict(prediction='{"result": 5}', verdict='{"verdict": "accept", "result": 5}',
                        prediction_finish="stop", verdict_finish="stop", signature="f() -> int",
                        expected=5, candidate=5, revised=False, selected="prediction",
                        answer_status=ParseStatus.PARSED, checker_status=ParseStatus.PARSED)
        cases = (
            dict(label="accepted"),
            dict(label="revised", prediction='{"result": 6}', candidate=6,
                 verdict='{"verdict": "revise", "result": 5}', revised=True, selected="checker"),
            dict(label="equal replacement", verdict='{"verdict": "revise", "result": 5}', selected="checker"),
            dict(label="malformed check", verdict="invalid verdict", checker_status=ParseStatus.FAILED),
            dict(label="truncated check", verdict='{"verdict": "revise", "result": 999}',
                 verdict_finish="length", checker_status=ParseStatus.FAILED),
            dict(label="recovered truncated prediction", prediction_finish="length", candidate=None,
                 verdict='{"verdict": "revise", "result": 5}', revised=True, selected="checker"),
            dict(label="accepted failed prediction", prediction_finish="length", candidate=None,
                 expected=None, answer_status=ParseStatus.FAILED),
            dict(label="explicit None recovery", prediction="invalid prediction", candidate=None,
                 verdict='{"verdict": "revise", "result": null}', signature="f() -> None",
                 expected=None, selected="checker"),
            dict(label="fallback replacement", prediction='{"result": "old"}', candidate="old",
                 verdict='{"verdict": "revise", "result": 7?}', signature="f() -> str",
                 expected="7?", revised=True, selected="checker", answer_status=ParseStatus.FALLBACK,
                 checker_status=ParseStatus.FALLBACK),
        )
        for mode in ("strict", "warn", "none"):
            for overrides in cases:
                case = defaults | overrides
                with self.subTest(mode=mode, case=case["label"]):
                    prediction_reason = "max_tokens" if case["prediction_finish"] == "length" else "prediction_end"
                    checker_reason = "max_output_tokens" if case["verdict_finish"] == "length" else "checker_end"
                    backend = ContractBackend((case["prediction"], case["prediction_finish"], prediction_reason),
                                              (case["verdict"], case["verdict_finish"], checker_reason))
                    invocation = request(case["signature"])
                    with VirtualRuntime(backend=backend, strategy="contract", validation_mode=mode) as runtime:
                        result = runtime.execute(invocation)
                    failed = case["answer_status"] is ParseStatus.FAILED
                    self.assertEqual(result.result, case["expected"])
                    self.assertEqual((result.revised, result.pre_revision_value), (case["revised"], case["candidate"]))
                    self.assertIs(result.parse_status, case["answer_status"])
                    self.assertIs(result.verdict_parse_status, case["checker_status"])
                    self.assertEqual(bool(result.parse_error), case["answer_status"] is not ParseStatus.PARSED)
                    self.assertEqual(bool(result.verdict_parse_error), case["checker_status"] is not ParseStatus.PARSED)
                    self.assertEqual(result.provider_finish_reason,
                                     checker_reason if case["selected"] == "checker" else prediction_reason)
                    self.assertEqual(result.verdict_finish_reason, checker_reason)
                    self.assertEqual(result.failure_type, "parse_failure" if failed else None)
                    self.assertEqual(result.validation_status, "unparsed" if failed else
                                     "not_checked" if mode == "none" else "valid")
                    self.assertEqual((len(backend.calls), result.model_calls, result.retries), (2, 2, 0))
                    self.assertEqual((result.input_tokens, result.output_tokens, result.total_tokens), (6, 10, 16))
                    self.assertIn(case["prediction"], result.raw_text)
                    self.assertIn(case["verdict"], result.raw_text)

                    row = build_row(result, run_id="offline", benchmark="humaneval_plus", item_id="example",
                                    signature=invocation.signature, docstring=invocation.docstring, inputs={},
                                    expected=case["expected"], correct=not failed)
                    self.assertEqual((row["correct"], row["scorable"], row["failure_type"]),
                                     (not failed, True, "parse_failure" if failed else ""))
                    wire = json.loads(json.dumps(asdict(result)))
                    stream = io.StringIO(newline="")
                    writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
                    writer.writeheader()
                    writer.writerow(row)
                    csv_row = next(csv.DictReader(io.StringIO(stream.getvalue())))
                    for field in ("provider_finish_reason", "verdict_parse_status", "verdict_parse_error", "verdict_finish_reason"):
                        self.assertEqual(row[field], wire[field] or "")
                        self.assertEqual(csv_row[field], wire[field] or "")

    def test_checker_diagnostics_are_retained_by_cache_hits_and_tracing(self):
        backend = ContractBackend(('{"result": 5}', "stop", "prediction_end"),
                                  ('{"verdict": "revise", "result": 999}', "length", "max_output_tokens"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            with VirtualRuntime(backend=backend, strategy="contract", cache_size=8, trace_file=path) as runtime:
                first = runtime.execute(request())
                cached = runtime.execute(request())
            entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(backend.calls), 2)
        self.assertFalse(first.cached)
        self.assertTrue(cached.cached)
        self.assertEqual((cached.model_calls, cached.total_tokens), (0, 0))
        self.assertIs(cached.verdict_parse_status, ParseStatus.FAILED)
        self.assertEqual(cached.verdict_parse_error, first.verdict_parse_error)
        self.assertIn("truncated", cached.verdict_parse_error)
        for entry in entries:
            self.assertEqual((entry["result"], entry["provider_finish_reason"]), (5, "prediction_end"))
            self.assertEqual((entry["verdict_parse_status"], entry["verdict_finish_reason"]),
                             ("failed", "max_output_tokens"))
            self.assertEqual(entry["verdict_parse_error"], first.verdict_parse_error)
        self.assertEqual([entry["cached"] for entry in entries], [False, True])


if __name__ == "__main__":
    unittest.main()
