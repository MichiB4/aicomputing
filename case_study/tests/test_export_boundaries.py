"""Span and export regressions independent of any model provider."""
import csv
import json

import pytest

from paper_parser import model, study
from paper_parser.contract import Extraction, Sentence
from paper_parser.dataset import load_split


@pytest.mark.parametrize("offsets", [
    ((2, 3), (0, 1)),
    ((0, 1), (0, 1)),
    ((-1, 3), (0, 1)),
    ((False, 1), (2, 3)),
    ((0, 1), (2, 4)),
    ((0, 1), (2,)),
])
def test_sentence_rejects_invalid_and_overlapping_token_offsets(offsets):
    with pytest.raises(ValueError, match="Token offsets"):
        Sentence("aba", ("a", "a"), offsets)


def test_invalid_unicode_model_text_does_not_prevent_checkpointing(tmp_path):
    raw_text = "invalid response: \ud800"
    outcome = Extraction("model", "invalid_output", None, 0, prediction={"raw_text": raw_text})
    record = study.make_record(load_split("test")[0], outcome, model.Settings())
    base = tmp_path / "unicode"
    study.save({"records": [record]}, base)
    document = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    assert document["records"][0]["outcome"]["prediction"]["raw_text"] == raw_text
    with base.with_suffix(".csv").open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["raw_text"] == "invalid response: \\ud800"
    assert json.loads(row["details"]) == document["records"][0]
