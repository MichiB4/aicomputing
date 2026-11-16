"""Accuracy policy belongs to evaluation and must preserve execution evidence."""
from dataclasses import asdict
import unittest

from aicomputing.request import ExecuteResult, FailureType
from evaluation.benchmarks.result_schema import build_row, is_scorable


def invocation(value=7, **settings):
    defaults = dict(result=value, raw_text="raw response", model="offline", strategy="direct",
                    parse_status="parsed", validation_status="not_checked")
    return ExecuteResult(**(defaults | settings))


def assess(result, *, correct=False, expected=7, signature="f(x: int) -> int"):
    return build_row(result, run_id="offline", benchmark="humaneval_plus", item_id="example",
                     signature=signature, docstring="Return the required value.", inputs={"x": 0},
                     expected=expected, correct=correct)


class EvaluationOutcomeTests(unittest.TestCase):
    def test_worker_values_are_assessed_without_prediction_parsing_metadata(self):
        result = invocation(None, parse_status=None, execution_details={"status": "ok", "value": None})
        row = assess(result, correct=True, expected=None, signature="f(x: int) -> None")
        self.assertEqual((row["correct"], row["structural_validity"], row["parse_status"]), (True, "valid", ""))
        self.assertEqual((row["parse_error"], row["execution_error"]), ("", ""))
        result = invocation([7], parse_status=None, execution_details={"status": "ok", "value": [7]})
        row = assess(result)
        self.assertEqual(row["failure_type"], "structural_validation_failure")

    def test_execution_errors_have_a_separate_csv_field(self):
        result = invocation(None, failure_type=FailureType.EXECUTION, validation_status="execution_failed",
                            parse_status=None, execution_error="runtime_error: ZeroDivisionError",
                            execution_details={"status": "runtime_error"})
        row = assess(result)
        self.assertEqual(row["execution_error"], result.execution_error)
        self.assertEqual((row["parse_error"], row["parse_status"]), ("", ""))
        self.assertEqual((row["correct"], row["scorable"], row["failure_type"]), (False, True, "code_runtime_error"))

    def test_success_and_wrong_answers_have_evaluator_owned_outcomes(self):
        success = assess(invocation(), correct=True)
        wrong = assess(invocation(8))
        self.assertEqual((success["correct"], success["scorable"], success["failure_type"]), (True, True, ""))
        self.assertEqual((wrong["correct"], wrong["scorable"], wrong["failure_type"]),
                         (False, True, "semantic_wrong_answer"))

    def test_infrastructure_is_excluded_even_when_comparison_would_match_none(self):
        for failure in (FailureType.TIMEOUT, FailureType.PROVIDER_ERROR, FailureType.HARNESS_ERROR):
            with self.subTest(failure=failure):
                result = invocation(None, failure_type=failure, validation_status="unavailable", parse_status=None)
                row = assess(result, correct=True, expected=None)
                self.assertEqual((row["correct"], row["scorable"], row["failure_type"]), (False, False, failure))
                self.assertIsNone(row["input_tokens"])
                self.assertEqual(row["structural_validity"], "not_assessed")

    def test_output_failures_are_incorrect_and_scorable(self):
        cases = ((FailureType.PARSE, "unparsed"),
                 (FailureType.STRUCTURAL_VALIDATION, "invalid_type"),
                 (FailureType.CONSTRAINT_VALIDATION, "invalid_constraint"))
        for failure, status in cases:
            with self.subTest(failure=failure):
                row = assess(invocation(None, failure_type=failure, validation_status=status), correct=True)
                self.assertEqual((row["correct"], row["scorable"], row["failure_type"]), (False, True, failure))

    def test_worker_timeouts_and_other_code_errors_are_output_failures(self):
        for status in ("timeout", "format_error", "syntax_error", "runtime_error", "serialization_error"):
            for failure in (FailureType.EXECUTION, FailureType.PARSE, None):
                with self.subTest(status=status, failure=failure):
                    result = invocation(None, failure_type=failure, validation_status="execution_failed",
                                        parse_status="failed", execution_details={"status": status})
                    row = assess(result, correct=True)
                    self.assertEqual((row["correct"], row["scorable"], row["failure_type"]),
                                     (False, True, "code_" + status))
                    self.assertEqual(result.failure_type, failure)
                    self.assertTrue(is_scorable(row["failure_type"]))
                    self.assertEqual(row["structural_validity"], "not_assessed")

    def test_broad_execution_label_requires_failed_worker_diagnostics(self):
        for execution in (None, {"status": "ok"}):
            with self.subTest(execution=execution), self.assertRaisesRegex(ValueError, "failed worker diagnostics"):
                assess(invocation(None, failure_type=FailureType.EXECUTION, execution_details=execution))
        with self.assertRaises(ValueError):
            is_scorable(FailureType.EXECUTION)

    def test_worker_harness_failure_is_infrastructure(self):
        result = invocation(None, failure_type=FailureType.EXECUTION, validation_status="execution_failed", parse_status="failed",
                            execution_details={"status": "harness_error"})
        row = assess(result, correct=True, expected=None)
        self.assertEqual((row["correct"], row["scorable"], row["failure_type"]),
                         (False, False, "harness_error"))

    def test_structural_and_reference_type_diagnostics_remain_separate(self):
        structural = assess(invocation("7?"))
        reference = assess(invocation("7?"), signature="f(x: int)")
        self.assertEqual(structural["failure_type"], "structural_validation_failure")
        self.assertEqual(reference["failure_type"], "reference_type_mismatch")
        self.assertTrue(structural["scorable"] and reference["scorable"])

    def test_classification_does_not_mutate_package_results(self):
        result = invocation("7?", parse_status="fallback_parsed", parse_error="preserved text")
        before = asdict(result)
        row = assess(result)
        self.assertEqual(asdict(result), before)
        self.assertIsNone(result.failure_type)
        self.assertEqual(row["failure_type"], "structural_validation_failure")
        self.assertEqual(row["raw_text"], "raw response")

    def test_unknown_failure_labels_cannot_silently_change_the_denominator(self):
        with self.assertRaises(ValueError):
            assess(invocation(None, failure_type="new_unclassified_failure"))
        with self.assertRaises(ValueError):
            assess(invocation(None, execution_details={"status": "new_worker_status"}))
        with self.assertRaises(ValueError):
            assess(invocation(None, failure_type=FailureType.EXECUTION, execution_details={"status": "new_worker_status"}))
        with self.assertRaises(ValueError):
            assess(invocation(None, failure_type="new_unclassified_failure", execution_details={"status": "runtime_error"}))
        for invalid in (False, 0, []):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                is_scorable(invalid)


if __name__ == "__main__":
    unittest.main()
