"""Offline, versioned benchmark data with an explicit review gate.

This is experiment infrastructure, independent of the oracle package. A release
contains data, not executable validators or reference implementations. Approval
is a recorded curation decision; schema validation cannot establish semantics.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from mbpp_ground_truth import decode_value, encode_value

FORMAT_VERSION = 1
BENCHMARKS = ("humaneval_plus", "mbpp_plus", "gsm8k")
PROFILES = {"structured_abs_1e-6", "numeric_abs_1e-3", "unordered_set_abs_1e-6",
            "mbpp_exact_or_float_v1", "mbpp_float_atol_1e-4_v1"}


def input_digest(inputs: dict) -> str:
    value = encode_value({name: inputs[name] for name in sorted(inputs)})
    raw = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate release field: {name}")
        result[name] = value
    return result


def load_release(path, *, expected_sha256=None, require_approved=True):
    """Read and validate exact release bytes; never fall back to raw datasets."""
    source = Path(path).resolve(strict=True)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Prepared dataset release SHA256 changed; restore the frozen release")
    data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    validate_release(data, require_approved=require_approved)
    return data, {"path": str(source), "sha256": digest, "release_id": data["release_id"]}


def validate_release(data, *, require_approved=True):
    if not isinstance(data, dict) or type(data.get("format_version")) is not int or data["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported prepared dataset format_version")
    if not isinstance(data.get("release_id"), str) or not data["release_id"].strip():
        raise ValueError("Prepared dataset requires a release_id")
    if data.get("status") not in ("in_review", "approved"):
        raise ValueError("Unknown prepared dataset review status")
    if require_approved and data["status"] != "approved":
        raise ValueError("Dataset preparation is incomplete: release is not approved for selection")
    policy = data.get("policy", {})
    if (policy.get("public_examples") != "retained_actual_input_excluded"
            or policy.get("model_outputs_used_for_curation") is not False):
        raise ValueError("Release must retain examples and exclude performance-based curation")
    benches = data.get("benchmarks")
    if not isinstance(benches, dict) or not benches or set(benches) - set(BENCHMARKS):
        raise ValueError("Invalid prepared benchmark inventory")
    for benchmark, inventory in benches.items():
        items = inventory.get("items")
        if not isinstance(items, list) or type(inventory.get("source_count")) is not int or inventory["source_count"] != len(items):
            raise ValueError(f"{benchmark}: incomplete source inventory")
        seen = set()
        for item in items:
            item_id = item.get("item_id")
            if not isinstance(item_id, str) or not item_id or item_id in seen:
                raise ValueError(f"{benchmark}: duplicate or invalid item ID")
            seen.add(item_id)
            if item.get("review_status") not in ("approved", "quarantined"):
                raise ValueError(f"{item_id}: unrecorded review decision")
            if item["review_status"] == "quarantined":
                if not item.get("reason"):
                    raise ValueError(f"{item_id}: quarantine requires a reason")
                continue
            _validate_approved_item(benchmark, item)


def _validate_approved_item(benchmark, item):
    from aicomputing.types import parse_signature
    from aicomputing.parsing import parse_result_detailed, ParseOutcome
    from aicomputing.prompting import _format_prompt_value
    from aicomputing.validation import validate_inputs, validate_output

    item_id = item["item_id"]
    for field in ("signature", "specification", "review_note"):
        if not isinstance(item.get(field), str) or not item[field].strip():
            raise ValueError(f"{item_id}: missing {field}")
    if not isinstance(item.get("evidence"), list) or not item["evidence"]:
        raise ValueError(f"{item_id}: missing semantic review evidence")
    profile = item.get("scoring_profile")
    if profile not in PROFILES:
        raise ValueError(f"{item_id}: unknown scoring profile")
    allowed_profiles = {
        "gsm8k": {"numeric_abs_1e-3"},
        "humaneval_plus": {"structured_abs_1e-6"},
        "mbpp_plus": {"mbpp_exact_or_float_v1", "mbpp_float_atol_1e-4_v1",
                      "unordered_set_abs_1e-6"},
    }
    if profile not in allowed_profiles[benchmark]:
        raise ValueError(f"{item_id}: scoring profile does not match benchmark")
    signature = parse_signature(item["signature"])
    observations = item.get("observations")
    examples = item.get("public_example_input_sha256")
    if not isinstance(examples, list) or any(not isinstance(value, str) or len(value) != 64 for value in examples):
        raise ValueError(f"{item_id}: missing example coverage ledger")
    if not isinstance(observations, list) or not observations:
        raise ValueError(f"{item_id}: no approved observations")
    ids, inputs_seen = set(), {}
    for observation in observations:
        key = observation.get("invocation_id")
        if not isinstance(key, str) or not key or key in ids:
            raise ValueError(f"{item_id}: duplicate or invalid invocation ID")
        ids.add(key)
        inputs = decode_value(observation["inputs"])
        expected = decode_value(observation["expected"])
        if type(inputs) is not dict or any(type(name) is not str for name in inputs):
            raise ValueError(f"{item_id}: inputs must be named Python values")
        if set(inputs) != {parameter.name for parameter in signature.params}:
            raise ValueError(f"{item_id}: inputs disagree with declared signature")
        validate_inputs(signature, inputs, strict=True)
        digest = input_digest(inputs)
        if digest in examples:
            raise ValueError(f"{item_id}: actual input repeats a displayed public example")
        encoded = encode_value(expected)
        if digest in inputs_seen and inputs_seen[digest] != encoded:
            raise ValueError(f"{item_id}: identical inputs have conflicting targets")
        inputs_seen[digest] = encoded
        # Exercise the actual parser/validator rather than inferring a return
        # type from the gold answer. Gold values remain grading-only data.
        parsed = parse_result_detailed(_format_prompt_value({"result": expected}),
                                       signature.return_type, strict=True)
        validate_output(parsed.value, signature.return_type, strict=True)
        if parsed.status == ParseOutcome.FAILED:
            raise ValueError(f"{item_id}: gold output cannot pass the runtime interface")
        if not compare_prepared(parsed.value, expected, profile):
            raise ValueError(f"{item_id}: gold output does not roundtrip through scoring")


def approved_items(data, benchmark):
    if benchmark not in data["benchmarks"]:
        raise ValueError(f"{benchmark}: absent from prepared release")
    return [item for item in data["benchmarks"][benchmark]["items"]
            if item["review_status"] == "approved"]


def release_for_manifest(manifest):
    reference = manifest.get("dataset_release")
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "release_id"}:
        raise ValueError("Reviewed experiment requires a frozen dataset_release reference")
    data, actual = load_release(reference["path"], expected_sha256=reference["sha256"])
    if reference["release_id"] != actual["release_id"]:
        raise ValueError("Prepared dataset release ID changed")
    return data


def compare_prepared(predicted: Any, expected: Any, profile: str) -> bool:
    if profile == "numeric_abs_1e-3":
        from scoring import answers_match
        return answers_match(predicted, expected)
    from scoring import results_match
    if profile in ("mbpp_exact_or_float_v1", "mbpp_float_atol_1e-4_v1"):
        return _mbpp_float_compare(predicted, expected,
                                   1e-4 if profile == "mbpp_float_atol_1e-4_v1" else 1e-6)
    if profile == "structured_abs_1e-6":
        return results_match(predicted, expected)
    if profile == "unordered_set_abs_1e-6":
        # Only tasks whose published scorer specifies set equivalence may
        # request this. Order and duplicate count are intentionally ignored.
        if type(predicted) not in (list, tuple, set, frozenset) or type(expected) not in (list, tuple, set, frozenset):
            return False
        try:
            return results_match(set(predicted), set(expected))
        except TypeError:
            return False
    raise ValueError(f"Unknown prepared scoring profile: {profile}")


def _mbpp_float_compare(predicted, expected, atol):
    """Published MBPP float tolerances, with strict integer and boolean identity.

    Approximation applies to float targets; integer labels remain exact.
    Sequence ordering and duplicate counts are preserved. Set-equivalence tasks
    use their separate profile. This does not execute the upstream scorer.
    """
    if type(expected) is float:
        if type(predicted) not in (int, float):
            return False
        try:
            return math.isfinite(predicted) and math.isfinite(expected) and math.isclose(
                predicted, expected, rel_tol=1e-7, abs_tol=atol)
        except OverflowError:
            return False
    if type(expected) in (list, tuple):
        return (type(predicted) in (list, tuple) and len(predicted) == len(expected)
                and all(_mbpp_float_compare(left, right, atol)
                        for left, right in zip(predicted, expected)))
    if type(expected) is dict:
        if type(predicted) is not dict or len(predicted) != len(expected):
            return False
        # Keys retain their original typed identity; approximation belongs to
        # numeric values, not dictionary key selection.
        try:
            keys = {json.dumps(encode_value(key), sort_keys=True): key for key in predicted}
        except (TypeError, ValueError):
            return False
        return all((encoded := json.dumps(encode_value(key), sort_keys=True)) in keys
                   and _mbpp_float_compare(predicted[keys[encoded]], value, atol)
                   for key, value in expected.items())
    if type(expected) is int:
        return type(predicted) in (int, float) and predicted == expected
    from scoring import results_match
    return results_match(predicted, expected)
