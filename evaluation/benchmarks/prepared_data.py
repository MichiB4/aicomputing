"""Offline, versioned benchmark data with an explicit review gate.

This is experiment infrastructure, independent of the oracle package. A release
contains data, not executable validators or reference implementations. Approval
is a recorded curation decision; schema validation cannot establish semantics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aicomputing.parsing import parse_result
from aicomputing.status import ParseStatus
from aicomputing.answer_protocol import frame_answer
from aicomputing.prompting import format_prompt_value
from aicomputing.types import parse_signature
from aicomputing.validation import validate_inputs, validate_output

from . import BENCHMARKS
from .scoring import PROFILES, compare
from .values import canonical_json, decode_value, encode_value

FORMAT_VERSION = 1

# Reviewed specifications wrap the original public text for prediction prompts.
REVIEWED_SPEC_PREFIX = (
    "The following is the original public task specification. Any calls, "
    "input/output pairs, or assertions inside it are public examples, not the "
    "actual input to solve.\n\n"
)
PREDICTION_SUFFIX = (
    "\n\nPredict only the return value for the concrete Inputs (Python values) "
    "in the actual task. Do not answer a public example."
)


def code_task_specification(specification: str) -> str:
    """Remove only the reviewed release's outer prediction wrapper, for code generation.

    Public specification/example blocks remain byte-for-byte identical. Other
    specifications, including similar wording inside public text, pass through.
    """
    if specification.startswith(REVIEWED_SPEC_PREFIX) and specification.endswith(PREDICTION_SUFFIX):
        public = specification[len(REVIEWED_SPEC_PREFIX):-len(PREDICTION_SUFFIX)]
        if (public.startswith("BEGIN ORIGINAL PUBLIC TASK SPECIFICATION\n")
                and "\nEND ORIGINAL PUBLIC TASK SPECIFICATION" in public):
            return public
    return specification


def input_digest(inputs: dict) -> str:
    raw = canonical_json({name: inputs[name] for name in sorted(inputs)})
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
        validate_inputs(signature, inputs)
        digest = input_digest(inputs)
        if digest in examples:
            raise ValueError(f"{item_id}: actual input repeats a displayed public example")
        encoded = encode_value(expected)
        if digest in inputs_seen and inputs_seen[digest] != encoded:
            raise ValueError(f"{item_id}: identical inputs have conflicting targets")
        inputs_seen[digest] = encoded
        # Exercise the actual parser/validator rather than inferring a return
        # type from the gold answer. Gold values remain grading-only data.
        parsed = parse_result(frame_answer(format_prompt_value({"result": expected})))
        validate_output(parsed.value, signature.return_type, strict=True)
        if parsed.status == ParseStatus.FAILED:
            raise ValueError(f"{item_id}: gold output cannot pass the runtime interface")
        if not compare(parsed.value, expected, profile):
            raise ValueError(f"{item_id}: gold output does not roundtrip through scoring")


def approved_items(data, benchmark):
    if benchmark not in data["benchmarks"]:
        raise ValueError(f"{benchmark}: absent from prepared release")
    return [item for item in data["benchmarks"][benchmark]["items"]
            if item["review_status"] == "approved"]
