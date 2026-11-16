import csv
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from aicomputing import VirtualRuntime
from paper_parser import model, study
from paper_parser.contract import EntityRecord
from paper_parser.dataset import Case, load_split
from test_paper_extraction import SENTENCE, SPANS, StubBackend, TEXT


def test_offline_study_creates_matching_csv_json_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(model.Settings, "runtime", lambda self: pytest.fail("Offline mode reached model"))
    output = tmp_path / "rules"
    assert study.main(["--sample-size", "3", "--seed", "7", "--output", str(output)]) == 0
    assert sorted(path.suffix for path in tmp_path.iterdir()) == [".csv", ".json"]
    document = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    with output.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(document["records"]) == 3
    assert all(json.loads(row["details"]) == entry for row, entry in zip(rows, document["records"], strict=True))
    assert document["status"] == "completed"
    assert document["summary"]["rules"]["sentences"] == 3
    original = output.with_suffix(".json").read_bytes()
    with pytest.raises(SystemExit):
        study.main(["--sample-size", "3", "--seed", "7", "--output", str(output)])
    assert original == output.with_suffix(".json").read_bytes()


def test_selection_is_reproducible_and_keeps_test_labels_out_of_baseline():
    first = study.select_cases("test", 10, 42, None)
    second = study.select_cases("test", 10, 42, None)
    assert [case.id for case in first] == [case.id for case in second]
    assert set(first).issubset(set(load_split("test")))


def test_paired_study_uses_identical_tokens_and_keeps_gold_out_of_prompt(tmp_path, monkeypatch):
    backend = StubBackend()
    gold = EntityRecord.from_dict(SPANS, SENTENCE)
    case = Case("T0001", "test", SENTENCE, gold)
    monkeypatch.setattr(study, "select_cases", lambda *args: [case])
    monkeypatch.setattr(model.Settings, "has_credentials", lambda self: True)
    monkeypatch.setattr(model.Settings, "runtime", lambda self: VirtualRuntime(backend=backend))
    output = tmp_path / "paired"
    assert study.main(["--engine", "both", "--output", str(output)]) == 0
    document = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert [row["outcome"]["engine"] for row in document["records"]] == ["rules", "model"]
    assert len({tuple(row["tokens"]) for row in document["records"]}) == 1
    assert document["records"][1]["grading"]["exact_sentence"]
    assert "gold" not in backend.calls[0][0].lower()
    assert document["records"][1]["outcome"]["prediction"]["raw_text"]


def test_transport_error_stops_study_and_counts_missed_spans(tmp_path, monkeypatch):
    from aicomputing.backends.base import BackendTimeoutError
    backend = StubBackend(error=BackendTimeoutError("SECRET_SENTINEL"))
    monkeypatch.setattr(model.Settings, "has_credentials", lambda self: True)
    monkeypatch.setattr(model.Settings, "runtime", lambda self: VirtualRuntime(backend=backend))
    output = tmp_path / "timeout"
    assert study.main(["--engine", "model", "--sample-size", "3", "--output", str(output)]) == 2
    document = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert document["status"] == "stopped" and len(document["records"]) == 1
    assert document["summary"]["model"]["failed_attempts"] == 1
    assert not document["records"][0]["grading"]["exact_sentence"]
    assert "SECRET_SENTINEL" not in json.dumps(document)


def test_text_only_ui_offline_and_model_paths(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PAPER_PARSER_KEY_FILE", raising=False)
    app = Path(__file__).resolve().parents[1] / "app.py"
    test = AppTest.from_file(str(app), default_timeout=20).run()
    assert not test.exception
    assert len(test.text_area) == 1
    assert test.button[0].disabled and test.button[1].disabled
    test.text_area[0].set_value(TEXT).run()
    test.button[0].click().run()
    assert not test.exception
    assert test.session_state["tagger_results"]["rules"]["outcome"]["status"] == "completed"

    monkeypatch.setattr(model.Settings, "has_credentials", lambda self: True)
    monkeypatch.setattr(model.Settings, "runtime", lambda self: VirtualRuntime(backend=StubBackend()))
    test.run()
    test.button[1].click().run()
    assert not test.exception
    assert set(test.session_state["tagger_results"]) == {"rules", "model"}
