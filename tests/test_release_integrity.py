"""Verify the distributed benchmark data, documentation and failure boundaries."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from aicomputing.answer_protocol import frame_answer
from aicomputing.parsing import parse_result
from evaluation.benchmarks.prepared_data import load_release, validate_release
from evaluation.config import configure

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def release():
    return load_release(ROOT / "data/dataset-release.json")[0]


def test_release_checksum_and_inventory_match_documentation(release):
    path = ROOT / "data/dataset-release.json"
    raw = path.read_bytes()
    guide = (ROOT / "data/README.md").read_text(encoding="utf-8")
    assert hashlib.sha256(raw).hexdigest() in guide
    assert f"{len(raw):,}" in guide
    assert release["release_id"] in guide
    assert {name: (len(inv["items"]), sum(i["review_status"] == "approved" for i in inv["items"]))
            for name, inv in release["benchmarks"].items()} == {
                "humaneval_plus": (164, 153), "mbpp_plus": (378, 343), "gsm8k": (1319, 1181)}


def test_frozen_task_content_has_its_recorded_digest(release):
    content = [
        (name, item["item_id"], item["review_status"], item.get("signature"), item.get("specification"),
         item.get("scoring_profile"), item.get("public_example_input_sha256"),
         [(ob["invocation_id"], ob.get("pool"), ob["inputs"], ob["expected"])
          for ob in item.get("observations", [])])
        for name, inv in release["benchmarks"].items() for item in inv["items"]
    ]
    raw = json.dumps(content, ensure_ascii=True, separators=(",", ":")).encode()
    expected = "873523b19b029b8c8a989e05397d8e9f428c98a87f4ab0e8449e04147128517b"
    assert hashlib.sha256(raw).hexdigest() == expected
    assert release["provenance"]["frozen_content_sha256"] == expected


def test_release_evidence_is_self_contained_and_targets_identify_sources(release):
    raw = json.dumps(release)
    assert ".publish" not in raw
    assert "C:\\" not in raw
    assert "previous-release.json" not in raw
    sources = [ob["source"] for item in release["benchmarks"]["mbpp_plus"]["items"]
               for ob in item.get("observations", [])]
    assert sum(source["kind"] == "published_literals" for source in sources) == 342
    reviewed = [source for source in sources if source["kind"] != "published_literals"]
    assert len(reviewed) == 1
    assert reviewed[0]["kind"] == "specification_reviewed_target"


@pytest.mark.parametrize("location,bad", [
    (("policy",), []),
    (("benchmarks", "gsm8k"), []),
    (("benchmarks", "gsm8k", "items", 0), None),
    (("benchmarks", "gsm8k", "items", 0, "evidence"), [None]),
    (("benchmarks", "gsm8k", "items", 0, "scoring_profile"), []),
    (("benchmarks", "gsm8k", "items", 0, "observations"), [None]),
    (("benchmarks", "gsm8k", "items", 0, "observations"), [{"invocation_id": "one"}]),
    (("benchmarks", "gsm8k", "items", 0, "observations", 0, "inputs"),
     ["dict", [[["str", "question"], ["int", "0x1"]]]]),
    (("benchmarks", "gsm8k", "items", 0, "observations", 0, "expected"), ["str", "not a number"]),
    (("benchmarks", "gsm8k", "items", 0, "public_example_input_sha256"), ["z" * 64]),
])
def test_malformed_releases_raise_configuration_errors(release, location, bad):
    altered = deepcopy(release)
    target = altered
    for key in location[:-1]:
        target = target[key]
    target[location[-1]] = bad
    with pytest.raises(ValueError):
        validate_release(altered)


@pytest.mark.parametrize("depth", [600, 1100])
def test_deeply_nested_model_answers_are_parse_failures(depth):
    text = frame_answer('{"result":' + "[" * depth + "0" + "]" * depth + "}")
    outcome = parse_result(text)
    assert outcome.status == "failed"
    assert outcome.value is None
    assert "nesting" in outcome.error


def test_sol_rejects_unsupported_reasoning_before_inference():
    with pytest.raises(ValueError, match="requires --reasoning low"):
        configure({"model": "gpt-6.1-sol"})
    settings, profile = configure({"model": "gpt-6.1-sol", "reasoning": "low"})
    assert settings["reasoning"] == "low"
    assert profile.api_format == "responses"
