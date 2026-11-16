"""Shared statuses keep wire values stable and reject misspelled labels."""

import csv
from dataclasses import asdict
import io
import json
import unittest

from aicomputing import ExecuteResult, ParseStatus, ValidationStatus
from aicomputing.cache import ResultCache
from aicomputing.parsing import ParseOutcome
from aicomputing.strategies.base import StrategyResult


def result(**settings):
    return ExecuteResult(**(dict(result=7, raw_text="reply", model="offline", strategy="direct") | settings))


class StatusTests(unittest.TestCase):
    def test_public_result_accepts_enum_or_valid_string_and_rejects_unknown_labels(self):
        for status in ValidationStatus:
            with self.subTest(validation=status):
                self.assertIs(result(validation_status=status.value).validation_status, status)
                self.assertIs(result(validation_status=status).validation_status, status)
        for status in ParseStatus:
            with self.subTest(parsing=status):
                self.assertIs(result(parse_status=status.value).parse_status, status)
                self.assertIs(result(parse_status=status).parse_status, status)
                self.assertIs(result(verdict_parse_status=status.value).verdict_parse_status, status)
                self.assertIs(result(verdict_parse_status=status).verdict_parse_status, status)
        self.assertIsNone(result().parse_status)
        self.assertIsNone(result().verdict_parse_status)
        for field in ("validation_status", "parse_status", "verdict_parse_status"):
            for value in ("typo", "", False, 1, []):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    result(**{field: value})

    def test_parser_and_strategy_results_use_the_same_status_definitions(self):
        self.assertIs(ParseOutcome(7, "parsed").status, ParseStatus.PARSED)
        self.assertIs(StrategyResult(7, "reply", parse_status="fallback_parsed").parse_status, ParseStatus.FALLBACK)
        self.assertIs(StrategyResult(7, "reply", verdict_parse_status="failed").verdict_parse_status, ParseStatus.FAILED)
        with self.assertRaises(ValueError):
            ParseOutcome(7, "typo")
        for field in ("parse_status", "verdict_parse_status"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                StrategyResult(7, "reply", **{field: "typo"})

    def test_json_csv_and_text_keep_the_existing_string_values(self):
        snapshot = result(validation_status="valid", parse_status="parsed")
        wire = json.loads(json.dumps(asdict(snapshot)))
        self.assertEqual((wire["validation_status"], wire["parse_status"]), ("valid", "parsed"))
        stream = io.StringIO(newline="")
        csv.writer(stream).writerow([snapshot.validation_status, snapshot.parse_status])
        self.assertEqual(stream.getvalue(), "valid,parsed\r\n")
        self.assertEqual(f"{snapshot.validation_status}: {snapshot.parse_status}", "valid: parsed")

    def test_cached_statuses_preserve_their_types_without_colliding_with_string_inputs(self):
        cache = ResultCache()
        snapshot = result(validation_status="valid", parse_status="parsed")
        settings = dict(signature="f(x) -> int", docstring="Return seven.", model="offline", strategy="direct")
        cache.put(inputs={"x": ParseStatus.PARSED}, result=snapshot, **settings)
        hit = cache.get(inputs={"x": ParseStatus.PARSED}, **settings)
        self.assertIs(hit.validation_status, ValidationStatus.VALID)
        self.assertIs(hit.parse_status, ParseStatus.PARSED)
        self.assertIsNone(cache.get(inputs={"x": "parsed"}, **settings))


if __name__ == "__main__":
    unittest.main()
