"""Offline checks of the published labels and the token-span contract."""
from __future__ import annotations

from collections import Counter
import json
from unittest.mock import Mock

import pytest

from aicomputing import VirtualRuntime
from aicomputing.backends.base import Backend, BackendTimeoutError, GenerateResponse
from paper_parser import conventional, model
from paper_parser.contract import EntityRecord, Sentence
from paper_parser.dataset import load_split

TEXT = "We evaluate sentiment classification on IMDB using F1."
SENTENCE = Sentence.from_text(TEXT)
SPANS = {"tasks": [{"start": 2, "end": 4}],
         "datasets": [{"start": 5, "end": 6}],
         "metrics": [{"start": 7, "end": 8}]}


class StubBackend(Backend):
    name = "offline"
    default_model = "offline"

    def __init__(self, value=None, error=None):
        self.value = SPANS if value is None else value
        self.error = error
        self.calls = []

    def generate(self, prompt, **options):
        self.calls.append((prompt, options))
        if self.error:
            raise self.error
        return GenerateResponse(text="BEGIN FINAL ANSWER\n" + json.dumps({"result": self.value})
                                + "\nEND FINAL ANSWER", model="offline", input_tokens=100, output_tokens=30)


def test_published_splits_and_entity_counts():
    train, test = load_split("train"), load_split("test")
    assert (len(train), len(test)) == (1523, 487)
    assert Counter(label for case in train for label, _, _ in case.gold.spans()) == {
        "TASK": 1219, "DATASET": 420, "METRIC": 536}
    assert Counter(label for case in test for label, _, _ in case.gold.spans()) == {
        "TASK": 396, "DATASET": 192, "METRIC": 174}
    for case in test:
        for _, start, end in case.gold.spans():
            assert case.sentence.span_text(start, end).strip()


def test_span_contract_checks_bounds_and_overlap():
    record = EntityRecord.from_dict(SPANS, SENTENCE)
    assert record.spans() == {("TASK", 2, 4), ("DATASET", 5, 6), ("METRIC", 7, 8)}
    assert record.to_dict()["datasets"][0]["text"] == "IMDB"
    with pytest.raises(ValueError, match="overlap"):
        EntityRecord.from_dict({**SPANS, "datasets": [{"start": 3, "end": 6}]}, SENTENCE)
    with pytest.raises(ValueError, match="valid token range"):
        EntityRecord.from_dict({**SPANS, "metrics": [{"start": 100, "end": 101}]}, SENTENCE)
    with pytest.raises(ValueError, match="exactly"):
        EntityRecord.from_dict({**SPANS, "scores": []}, SENTENCE)


def test_conventional_extractor_uses_phrase_matches_without_model():
    glossary = {("sentiment", "classification"): "TASK", ("imdb",): "DATASET", ("f1",): "METRIC"}
    outcome = conventional.extract(SENTENCE, lexicon=glossary)
    assert outcome.status == "completed" and outcome.record.spans() == EntityRecord.from_dict(SPANS, SENTENCE).spans()
    assert conventional.extract(SENTENCE).status == "completed"


def test_model_uses_public_runtime_without_gold_or_baseline(monkeypatch):
    monkeypatch.setattr(conventional, "extract", Mock(side_effect=AssertionError("Reached baseline")))
    backend = StubBackend()
    with VirtualRuntime(backend=backend) as runtime:
        outcome = model.extract(SENTENCE, runtime, model.Settings())
    assert outcome.status == "completed" and outcome.record.spans() == EntityRecord.from_dict(SPANS, SENTENCE).spans()
    assert len(backend.calls) == 1
    assert "expected" not in backend.calls[0][0]
    assert model.estimated_cost(outcome, model.Settings(input_price=1, output_price=5)) == .00025


def test_invalid_model_span_is_withheld_and_raw_response_kept():
    backend = StubBackend({**SPANS, "metrics": [{"start": 100, "end": 101}]})
    with VirtualRuntime(backend=backend) as runtime:
        outcome = model.extract(SENTENCE, runtime, model.Settings())
    assert outcome.status == "invalid_output" and outcome.record is None
    assert any("valid token range" in warning for warning in outcome.warnings)
    assert outcome.prediction["raw_text"]


def test_timeout_and_private_key_are_not_leaked(tmp_path, monkeypatch):
    backend = StubBackend(error=BackendTimeoutError("SECRET_SENTINEL"))
    with VirtualRuntime(backend=backend) as runtime:
        outcome = model.extract(SENTENCE, runtime, model.Settings())
    assert outcome.status == "timeout" and outcome.record is None
    assert "SECRET_SENTINEL" not in json.dumps(outcome.to_dict())
    key = tmp_path / "key.env"
    key.write_text('OPENAI_API_KEY="SECRET_SENTINEL"\n', encoding="utf-8")
    constructor = Mock()
    monkeypatch.setattr(model, "VirtualRuntime", constructor)
    settings = model.Settings(key_file=key)
    settings.runtime()
    assert constructor.call_args.kwargs["api_key"] == "SECRET_SENTINEL"
    assert "SECRET_SENTINEL" not in json.dumps(settings.public())
