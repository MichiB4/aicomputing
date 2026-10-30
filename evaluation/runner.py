"""Maintained entry point for future text-preserving Luna evaluations.

Run from the repository root with: python -m evaluation.runner --help

Each model request gets at most three attempts, retrying only a dispatched timeout
without a returned response. Each attempt keeps the 60-second HTTP phase timeout
and its own budget reservation. Parsing never triggers a retry. Contract retries
only its failed stage. Unknown usage stays unknown; reported partial usage and
all physical attempts remain in the ledger. Existing runs are never changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace

MODEL = "gpt-6-luna"
API_BASE = "https://api.openai.com/v1"
BENCHMARKS = ("humaneval_plus", "mbpp_plus", "gsm8k")
STRATEGIES = ("direct", "cot", "pseudocode", "contract")
INPUT_RATE = Decimal("0.10") / 1_000_000
CACHED_INPUT_RATE = Decimal("0.01") / 1_000_000
CACHE_WRITE_RATE = Decimal("0.125") / 1_000_000
OUTPUT_RATE = Decimal("0.50") / 1_000_000
OUTPUT_RESERVATION = 128_000
MAX_INPUT_RESERVATION = 272_000
FRAMING_RESERVE = 4096
FORBIDDEN_TOKEN_FIELDS = ("max_tokens", "max_completion_tokens", "max_output_tokens", "max_new_tokens")
RUNNER_PROTOCOL = "evaluation-luna-text-timeout-v2"
RUNNER_SCHEMA_VERSION = 7
ANSWER_FORMAT = "framed_text_v1"
TIMEOUT_RETRIES = 2
HTTP_PHASE_TIMEOUT_SECONDS = 60.0
SOURCE_ROOTS = ("aicomputing/src", "aicomputing/benchmarks", "evaluation")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(workspace):
    """Inventory maintained Python sources using portable repository paths."""
    hashes = {}
    for relative in SOURCE_ROOTS:
        source_root = workspace / relative
        if not source_root.is_dir():
            raise ValueError("Required source tree is missing: " + relative)
        for path in sorted(source_root.rglob("*.py")):
            hashes[path.relative_to(workspace).as_posix()] = sha256(path)
    hashes["aicomputing/pyproject.toml"] = sha256(workspace / "aicomputing/pyproject.toml")
    return hashes


def pricing_snapshot():
    return {
        "input_usd_per_million": format((INPUT_RATE * 1_000_000).normalize(), "f"),
        "cached_input_usd_per_million": format((CACHED_INPUT_RATE * 1_000_000).normalize(), "f"),
        "cache_write_usd_per_million": format((CACHE_WRITE_RATE * 1_000_000).normalize(), "f"),
        "output_usd_per_million": format((OUTPUT_RATE * 1_000_000).normalize(), "f"),
        "provider_output_reservation_tokens": OUTPUT_RESERVATION,
        "max_input_reservation_tokens": MAX_INPUT_RESERVATION,
        "framing_reserve": FRAMING_RESERVE,
        "price_source": "https://developers.openai.com/api/docs/models/gpt-6-luna",
        "price_checked_date": "2026-10-26",
    }


def frozen_budget(manifest):
    limits = manifest["experiment_protocol"].get("cost_guard")
    if not isinstance(limits, dict) or set(limits) != {"operational_usd", "user_ceiling_usd"}:
        raise ValueError("Manifest requires explicit frozen budget limits")
    if any(not isinstance(value, str) for value in limits.values()):
        raise ValueError("Frozen budget values must be decimal strings")
    budget, ceiling = Decimal(limits["operational_usd"]), Decimal(limits["user_ceiling_usd"])
    if not budget.is_finite() or not ceiling.is_finite() or not Decimal(0) < budget <= ceiling:
        raise ValueError("Frozen limits require finite 0 < budget <= user ceiling")
    return budget, ceiling


def validate_output_dir(path, workspace):
    """Allow a new run below runs/; never reuse a directory or write elsewhere."""
    output_dir = path.resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite or resume an existing output directory")
    runs_root = workspace.resolve() / "runs"
    if runs_root not in output_dir.parents:
        raise ValueError("Run outputs must be a new subdirectory of runs/")
    return output_dir


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_json(stream, value):
    stream.write(json.dumps(value, ensure_ascii=True, default=str) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def read_key(path):
    """Read one named credential without printing values or dotenv contents."""
    if path is None:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise ValueError("OPENAI_API_KEY is absent; provide --key-file or the environment variable")
        return key
    found = []
    for original in path.read_text(encoding="utf-8-sig").splitlines():
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if separator and name.strip() == "OPENAI_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if value and not any(character.isspace() for character in value):
                found.append(value)
    if len(found) != 1:
        raise ValueError("Credential file must contain exactly one nonempty OPENAI_API_KEY assignment")
    return found[0]


class StopRun(Exception):
    pass


class RecoverableInfrastructure(StopRun):
    """A dispatched transport failure after any permitted timeout retry."""


def validate_timeout(seconds):
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("HTTP phase timeout must be a finite positive number of seconds")
    return float(seconds)


def build_sdk_client(api_key, timeout_seconds):
    """Configure both timeout layers without performing an HTTP request."""
    import httpx
    from openai import OpenAI
    timeout = httpx.Timeout(validate_timeout(timeout_seconds))
    http_client = httpx.Client(follow_redirects=False, timeout=timeout)
    try:
        return OpenAI(api_key=api_key, base_url=API_BASE, max_retries=0,
                      http_client=http_client, timeout=timeout)
    except BaseException:
        http_client.close()
        raise


def sanitized_exception_diagnostics(exception):
    """Record class identities and HTTP phases; never serialize exception text."""
    import httpx
    chain, phases, seen = [], [], set()
    cause = exception
    for _ in range(12):
        if cause is None or id(cause) in seen:
            break
        seen.add(id(cause))
        qualified = type(cause).__module__ + "." + type(cause).__name__
        chain.append(qualified if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,255}", qualified, flags=re.ASCII) else "unrecorded_exception_class")
        for exception_class, phase in ((httpx.ConnectTimeout, "connect"), (httpx.ReadTimeout, "read"),
                                       (httpx.WriteTimeout, "write"), (httpx.PoolTimeout, "pool")):
            if isinstance(cause, exception_class) and phase not in phases:
                phases.append(phase)
        cause = cause.__cause__
    return {"exception_cause_chain_types": chain, "httpx_timeout_phases": phases,
            "timeout_phase_source": "httpx_exception_class" if phases else "unavailable"}


def is_transport_exception(exception):
    """Require a genuine SDK transport cause, not a statusless API wrapper."""
    from openai import APIConnectionError, APITimeoutError
    from aicomputing.backends.base import BackendTimeoutError, BackendCommunicationError
    if not isinstance(exception, (BackendTimeoutError, BackendCommunicationError)):
        return False
    cause = exception.__cause__
    for _ in range(6):
        if cause is None:
            return False
        if isinstance(cause, (APIConnectionError, APITimeoutError)):
            return True
        cause = cause.__cause__
    return False


def infrastructure_result(records, *, model, strategy):
    """Preserve incomplete call evidence without inventing a final answer."""
    from aicomputing.request import ExecuteResult, FailureType
    if not records or any(not record["sdk_dispatched"] for record in records):
        raise ValueError("Infrastructure outcome requires dispatched evidence")
    failure = FailureType.TIMEOUT if records[-1].get("exception_type") == "BackendTimeoutError" else FailureType.PROVIDER_ERROR
    prompts = [message["content"] for record in records for message in record["dispatched_messages"]
               if message.get("role") == "user"]
    responses = [record.get("response_text") for record in records if record.get("returned_model")]
    return ExecuteResult(result=None, raw_text="\n".join(text for text in responses if isinstance(text, str)),
                         model=model, strategy=strategy, scorable=False, failure_type=failure,
                         answer_format=ANSWER_FORMAT,
                         transport_diagnostics=[record.get("transport_diagnostics") for record in records],
                         validation_status="unavailable", parse_status=None, prompts=prompts,
                         input_tokens=None, output_tokens=None, total_tokens=None,
                         model_calls=len(records), retries=0, token_retries=0,
                         latency_ms=1000 * sum(record["elapsed_seconds"] for record in records),
                         reasoning_requests=[{"reasoning_effort": "none"} for record in records],
                         reasoning_observations=[({"reported_reasoning_tokens": record.get("reasoning_tokens"),
                                                  "returned_reasoning_chars": record.get("returned_reasoning_chars"),
                                                  "source": "provider_response"} if record.get("returned_model") else None)
                                                 for record in records])


def generate_audited_request(backend, messages, options, *, ledger, stream, call):
    """Retry timeout failures up to twice, never a completed response."""
    from aicomputing.backends.base import BackendTimeoutError
    if options.get("max_new_tokens") is not None or options.get("model", MODEL) not in (None, MODEL):
        raise StopRun("application_token_or_model_guard")
    backend._get_client()
    first_attempt = None
    for request_attempt in range(1, TIMEOUT_RETRIES + 2):
        backend._record, reservation, started = ledger.before(messages)
        first_attempt = first_attempt or backend._record["attempt"]
        backend._record.update(logical_request_first_attempt=first_attempt,
                               request_attempt=request_attempt)
        backend._sdk_response = None
        try:
            response = call(messages, **options)
        except BaseException as exc:
            ledger.after(backend._record, reservation, started,
                         sdk_response=backend._sdk_response, exception=exc)
            status, unsupported_none, identifiers = classify_exception(exc)
            recoverable = (is_transport_exception(exc) and backend._sdk_response is None
                           and backend._record["sdk_dispatched"] and backend._record["sdk_wire_checked"]
                           and status is None and not unsupported_none and not identifiers
                           and ledger.stop_reason == "missing_or_invalid_usage"
                           and backend._record["charge_basis"] == "reservation_unknown_usage")
            if not recoverable:
                raise StopRun(ledger.stop_reason or "provider_exception") from None
            retry = isinstance(exc, BackendTimeoutError) and request_attempt <= TIMEOUT_RETRIES
            append_json(stream, {"event": "transport_failure", "attempt": backend._record["attempt"],
                                 "logical_request_first_attempt": first_attempt,
                                 "request_attempt": request_attempt, "exception_type": type(exc).__name__,
                                 "retained_unknown_reservation_usd": backend._record["charge_usd"],
                                 "timeout_retry_scheduled": retry,
                                 "guard_cleared": "missing_or_invalid_usage"})
            ledger.stop_reason = None
            if retry:
                # before() rechecks budget and reserves this next physical request.
                continue
            raise RecoverableInfrastructure(type(exc).__name__) from None
        ledger.after(backend._record, reservation, started,
                     sdk_response=backend._sdk_response, response=response)
        if ledger.stop_reason:
            raise StopRun(ledger.stop_reason)
        return response


def account_attempts(result, records):
    """Use dispatched attempts, including unknown usage, as accounting evidence."""
    if not records or any(not record["sdk_dispatched"] for record in records):
        raise StopRun("evaluation_attempt_accounting_guard")
    complete = all(record["usage_complete"] for record in records)
    reported_input = sum(record.get("input_tokens") or 0 for record in records)
    reported_output = sum(record.get("output_tokens") or 0 for record in records)
    result.model_calls = len(records)
    result.input_tokens = reported_input if complete else None
    result.output_tokens = reported_output if complete else None
    result.total_tokens = reported_input + reported_output if complete else None
    result.tokens_used = reported_output if complete else None
    result.latency_ms = 1000 * sum(record["elapsed_seconds"] for record in records)
    result.reasoning_requests = [{"reasoning_effort": "none"} for record in records]
    result.reasoning_observations = [
        {"reported_reasoning_tokens": record.get("reasoning_tokens"),
         "returned_reasoning_chars": record.get("returned_reasoning_chars"),
         "source": "provider_response"} if record.get("returned_model") else None
        for record in records]
    result.transport_diagnostics = [
        {**(record.get("transport_diagnostics") or {}),
         "attempt": record["attempt"],
         "logical_request_first_attempt": record["logical_request_first_attempt"],
         "request_attempt": record["request_attempt"],
         "exception_type": record.get("exception_type")}
        for record in records]
    return {"retries": result.retries + sum(record["request_attempt"] > 1 for record in records)}


def is_token_count(value):
    return type(value) is int and value >= 0


def safe_sdk_response(response, tool_detector):
    """Allowlist successful SDK metadata before the production adapter parses it."""
    raw = response.model_dump() if hasattr(response, "model_dump") else {}
    raw = raw if isinstance(raw, dict) else {}
    choices = raw.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message")
    message = message if isinstance(message, dict) else {}
    usage = raw.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    prompt_details = usage.get("prompt_tokens_details")
    prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
    completion_details = usage.get("completion_tokens_details")
    completion_details = completion_details if isinstance(completion_details, dict) else {}
    sdk_choices = getattr(response, "choices", None)
    sdk_choice = sdk_choices[0] if sdk_choices else SimpleNamespace()
    evidence = tool_detector(sdk_choice, raw)
    returned_reasoning = message.get("reasoning", message.get("reasoning_content"))
    return {
        "response_id": raw.get("id"), "returned_model": raw.get("model"),
        "system_fingerprint": raw.get("system_fingerprint"),
        "service_tier": raw.get("service_tier"),
        "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
        "reported_total_tokens": usage.get("total_tokens"),
        "cached_input_tokens_reported": prompt_details.get("cached_tokens"),
        "cache_write_tokens_reported": prompt_details.get("cache_write_tokens"),
        "reasoning_tokens": completion_details.get("reasoning_tokens"),
        "finish_reason": choice.get("finish_reason"), "response_text": message.get("content"),
        "returned_reasoning_chars": len(returned_reasoning) if isinstance(returned_reasoning, str) else None,
        "tool_use_detected": evidence is not None, "tool_use_evidence": evidence,
    }


def classify_exception(exception):
    """Inspect provider errors in memory, retaining no error body or headers."""
    status = None
    unsupported_none = False
    identifiers = {}
    cause = exception
    for _ in range(6):
        candidate = getattr(cause, "status_code", None)
        if type(candidate) is int:
            status = candidate
        body = getattr(cause, "body", None)
        if isinstance(body, dict):
            error = body.get("error", body)
            if isinstance(error, dict):
                for key in ("code", "type", "param"):
                    value = error.get(key)
                    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value, flags=re.ASCII):
                        identifiers["provider_error_" + key] = value
                text = str(error.get("message", "")).lower()
                param = str(error.get("param", "")).lower()
                unsupported_none = unsupported_none or (
                    candidate == 400
                    and
                    ("reasoning_effort" in param or "reasoning_effort" in text)
                    and "none" in text
                    and any(word in text for word in ("unsupported", "not supported", "invalid", "supported values"))
                )
        cause = getattr(cause, "__cause__", None)
        if cause is None:
            break
    return status, unsupported_none, identifiers


class Ledger:
    def __init__(self, stream, limit, user_ceiling, max_attempts, timeout_seconds=HTTP_PHASE_TIMEOUT_SECONDS):
        self.stream, self.limit, self.user_ceiling = stream, limit, user_ceiling
        self.max_attempts = max_attempts
        self.timeout_seconds = validate_timeout(timeout_seconds)
        self.attempts = self.responses = 0
        self.charged = self.usage_based_charge = self.exact_reported_charge = self.unknown_reservations = Decimal(0)
        self.context, self.records = {}, []
        self.stop_reason = None

    def before(self, messages):
        if self.stop_reason or self.attempts >= self.max_attempts:
            self.stop_reason = self.stop_reason or "generation_attempt_cap"
            raise StopRun(self.stop_reason)
        input_bound = len(json.dumps(messages, ensure_ascii=True).encode("utf-8")) + FRAMING_RESERVE
        if input_bound > MAX_INPUT_RESERVATION:
            self.stop_reason = "long_context_financial_guard"
            raise StopRun(self.stop_reason)
        # Cache writes cost more than ordinary input. Reserve the higher rate.
        reservation = input_bound * CACHE_WRITE_RATE + OUTPUT_RESERVATION * OUTPUT_RATE
        if self.charged + reservation > self.limit or self.charged + reservation > self.user_ceiling:
            self.stop_reason = "budget_reservation_guard"
            raise StopRun(self.stop_reason)
        self.attempts += 1
        record = {
            "attempt": self.attempts, **self.context, "started_at_utc": utc_now(),
            "requested_model": MODEL, "application_max_new_tokens": None,
            "provider_output_reservation_tokens": OUTPUT_RESERVATION,
            "input_reservation_tokens": input_bound, "reservation_usd": str(reservation),
            "sdk_wire_checked": False, "sdk_dispatched": False,
            "dispatched_messages": messages, "prompt_provenance": "captured before dispatch",
            "http_phase_timeout_seconds": self.timeout_seconds,
        }
        append_json(self.stream, {"event": "request_start", **record})
        return record, reservation, time.perf_counter()

    def after(self, record, reservation, started, *, sdk_response=None, response=None, exception=None):
        record.update(elapsed_seconds=time.perf_counter() - started, finished_at_utc=utc_now(),
                      exception_type=type(exception).__name__ if exception else None)
        if sdk_response is not None:
            self.responses += 1
            record.update(sdk_response)
        if response is not None:
            record["normalized_finish_reason"] = response.finish_reason
            record["transport_diagnostics"] = response.transport_diagnostics
            response.observed_reasoning = {
                "reported_reasoning_tokens": record.get("reasoning_tokens"),
                "returned_reasoning_chars": record.get("returned_reasoning_chars"),
                "source": "provider_response",
            }
        input_tokens, output_tokens = record.get("input_tokens"), record.get("output_tokens")
        complete = is_token_count(input_tokens) and is_token_count(output_tokens)
        cached = record.get("cached_input_tokens_reported")
        cached_missing = cached is None
        cached_valid = cached_missing or (is_token_count(cached) and complete and cached <= input_tokens)
        standard_tier = record.get("service_tier") in (None, "default")
        cache_write = record.get("cache_write_tokens_reported")
        write_missing = cache_write is None
        write_valid = write_missing or (is_token_count(cache_write) and complete and cache_write <= input_tokens)
        if complete and cached_valid and write_valid:
            effective_cached = 0 if cached_missing else cached
            effective_write = input_tokens - effective_cached if write_missing else cache_write
            if effective_cached + effective_write > input_tokens:
                write_valid = False
        if complete and cached_valid and write_valid and standard_tier:
            charge = ((input_tokens - effective_cached - effective_write) * INPUT_RATE
                      + effective_cached * CACHED_INPUT_RATE + effective_write * CACHE_WRITE_RATE
                      + output_tokens * OUTPUT_RATE)
            self.usage_based_charge += charge
            if not cached_missing and not write_missing:
                self.exact_reported_charge += charge
            record.update(cached_input_tokens_accounted=effective_cached,
                          cache_write_tokens_accounted=effective_write,
                          cached_input_missing_assumed_zero=cached_missing,
                          cache_write_missing_assumed_all_uncached_write=write_missing,
                          charge_is_upper_bound=cached_missing or write_missing,
                          charge_basis="reported_usage_cache_upper_bound" if cached_missing or write_missing else "reported_usage")
            if input_tokens > record["input_reservation_tokens"] or output_tokens > OUTPUT_RESERVATION or charge > reservation:
                self.stop_reason = "provider_usage_exceeded_reservation"
        else:
            charge = reservation if record["sdk_dispatched"] else Decimal(0)
            self.unknown_reservations += charge
            record["charge_basis"] = ("reservation_unknown_service_tier_pricing" if not standard_tier
                                      else "reservation_unknown_usage") if record["sdk_dispatched"] else "no_dispatch"
            if not standard_tier:
                self.stop_reason = "service_tier_mismatch"
            elif not complete or (cached_valid and write_valid):
                self.stop_reason = "missing_or_invalid_usage"
            else:
                self.stop_reason = "invalid_cache_token_usage"
        if record.get("tool_use_detected"):
            self.stop_reason = "tool_use_detected"
        if sdk_response is not None and record.get("returned_model") != MODEL:
            self.stop_reason = "returned_model_mismatch"
        if record.get("finish_reason") == "length" or record.get("normalized_finish_reason") == "length":
            self.stop_reason = "provider_output_truncated"
        if not standard_tier:
            self.stop_reason = "service_tier_mismatch"
        if exception is not None:
            record["transport_diagnostics"] = getattr(exception, "diagnostics", None)
            record.update(sanitized_exception_diagnostics(exception))
            status, unsupported_none, identifiers = classify_exception(exception)
            record.update(identifiers)
            if status is not None:
                record["http_status_code"] = status
            record["reasoning_none_explicitly_unsupported"] = unsupported_none
            self.stop_reason = self.stop_reason or "provider_exception"
            if unsupported_none:
                self.stop_reason = "reasoning_effort_none_unsupported"
            elif isinstance(exception, StopRun) and not record["sdk_dispatched"]:
                self.stop_reason = str(exception)
        self.charged += charge
        record.update(usage_complete=complete and cached_valid and write_valid, charge_usd=str(charge),
                      cumulative_conservative_charge_usd=str(self.charged), stop_reason=self.stop_reason)
        self.records.append(record)
        append_json(self.stream, {"event": "request_complete", **record})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--budget-usd", type=Decimal)
    parser.add_argument("--user-ceiling-usd", type=Decimal)
    parser.add_argument("--http-timeout-seconds", type=float, default=HTTP_PHASE_TIMEOUT_SECONDS,
                        help="Uniform connect/read/write/pool phase timeout; not an overall request deadline (default: 60)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    timeout_seconds = validate_timeout(args.http_timeout_seconds)
    if timeout_seconds != HTTP_PHASE_TIMEOUT_SECONDS:
        parser.error("This protocol fixes the HTTP phase timeout at 60 seconds per attempt")
    started = time.perf_counter()
    workspace = Path(__file__).resolve().parents[1]
    for relative in ("aicomputing/src", "aicomputing/benchmarks"):
        sys.path.insert(0, str(workspace / relative))
    os.environ.update(HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      TOKENIZERS_PARALLELISM="false")
    from experiment import (resolve_generation_settings, resolve_model_connection,
                            resolve_reasoning_settings, verify_content)
    from evaluation.reproducibility import load_inputs
    prepared = load_inputs(args.manifest)
    manifest = prepared.manifest
    if manifest.get("data_policy") != "reviewed_release_v1":
        raise ValueError("This runner requires an approved, hash-pinned reviewed data release")
    experiment_protocol = manifest.get("experiment_protocol")
    if not isinstance(experiment_protocol, dict) or experiment_protocol.get("runner_protocol") != RUNNER_PROTOCOL:
        raise ValueError("Future runs require a newly rebuilt manifest explicitly tagged for this corrected protocol")
    if manifest.get("answer_format") != ANSWER_FORMAT:
        raise ValueError("This protocol requires answer_format=framed_text_v1")
    if type(experiment_protocol.get("timeout_retries")) is not int or experiment_protocol["timeout_retries"] != TIMEOUT_RETRIES:
        raise ValueError("Manifest must declare exactly two timeout retries per model request")
    actual_sources = source_hashes(workspace)
    if (experiment_protocol.get("source_sha256") != actual_sources
            or experiment_protocol.get("runner_sha256") != sha256(__file__)):
        raise ValueError("Prepared source or runner changed; freeze and verify a new protocol before inference")
    if ("uniform_http_phase_timeout_seconds" in experiment_protocol
            and validate_timeout(experiment_protocol["uniform_http_phase_timeout_seconds"]) != timeout_seconds):
        raise ValueError("Manifest-declared uniform timeout differs from the requested HTTP timeout")
    config = manifest["model_configs"][MODEL]
    generation = resolve_generation_settings(manifest.get("generation", {}), config)
    connection = resolve_model_connection(config)
    if (manifest["models"] != [MODEL] or manifest["strategies"] != list(STRATEGIES)
            or manifest.get("baseline") is not None or set(manifest["benchmarks"]) != set(BENCHMARKS)
            or manifest["validation_mode"] != "strict"):
        raise ValueError("Manifest must specify exactly the approved model, benchmarks, four strategies and strict validation")
    if generation != {"temperature": 0.0, "top_p": 1.0, "max_retries": 0, "max_new_tokens": None}:
        raise ValueError("Manifest must omit the output cap and disable parsing/validation retries")
    if resolve_reasoning_settings("openai", config.get("reasoning", {})) != {"reasoning_effort": "none"}:
        raise ValueError("This runner requires native reasoning effort none; fallback needs a separate reviewed runner")
    if connection != {"provider": None, "backend": "openai", "base_url": API_BASE}:
        raise ValueError("Only the official OpenAI endpoint is approved")
    sizes = {benchmark: len(manifest["benchmarks"][benchmark]["items"]) for benchmark in BENCHMARKS}
    if len(set(sizes.values())) != 1 or next(iter(sizes.values())) not in (5, 100):
        raise ValueError("Manifest must contain either five or 100 frozen cases per benchmark")
    cases_per_benchmark = next(iter(sizes.values()))
    for benchmark in BENCHMARKS:
        ids = [item["item_id"] for item in manifest["benchmarks"][benchmark]["items"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Each manifest item must represent a distinct coding function or question")
    planned_evaluations = 3 * cases_per_benchmark * 4
    max_attempts = 3 * cases_per_benchmark * 5 * (TIMEOUT_RETRIES + 1)
    fixed_settings = {"sdk_retries": 0, "validation_retries": 0, "application_output_cap": None,
                      "native_reasoning_effort": "none", "tools_supplied": False, "answer_format": ANSWER_FORMAT}
    if any(name not in experiment_protocol or type(experiment_protocol[name]) is not type(value)
           or experiment_protocol[name] != value for name, value in fixed_settings.items()):
        raise ValueError("Frozen request policy differs from the reviewed runner")
    if "uniform_http_phase_timeout_seconds" not in experiment_protocol:
        raise ValueError("Manifest must freeze the HTTP phase timeout")
    if experiment_protocol.get("pricing") != pricing_snapshot():
        raise ValueError("Frozen pricing differs from the reviewed accounting policy")
    budget, ceiling = frozen_budget(manifest)
    if ((args.budget_usd is not None and args.budget_usd != budget)
            or (args.user_ceiling_usd is not None and args.user_ceiling_usd != ceiling)):
        parser.error("CLI budget limits must match the frozen manifest")
    counts = {"evaluations": planned_evaluations, "nominal_model_calls": 3 * cases_per_benchmark * 5,
              "max_generation_attempts": max_attempts}
    if any(type(experiment_protocol.get(name)) is not int or experiment_protocol[name] != count
           for name, count in counts.items()):
        raise ValueError("Frozen plan counts differ from the selected cases")
    print(f"Plan verified: {3 * cases_per_benchmark} unique cases, {planned_evaluations} evaluations, at most {max_attempts} model calls; HTTP phase timeout {timeout_seconds:g}s; output cap omitted, native reasoning none, tools unavailable (not supplied).", flush=True)
    if args.dry_run:
        return 0
    from adapters import get_adapter
    resolved = {}
    for benchmark in BENCHMARKS:
        items = get_adapter(benchmark).resolve(manifest)
        verify_content(manifest, benchmark, items, hard=True)
        if len(items) != sizes[benchmark]:
            raise ValueError("Resolved case count differs from the frozen manifest")
        resolved[benchmark] = items
        print(f"Verified frozen offline content: {benchmark}, {len(items)} cases.", flush=True)
    if args.preflight_only:
        return 0
    if args.output_dir is None:
        parser.error("Live execution requires a new --output-dir; existing run directories are never reused")
    output_dir = validate_output_dir(args.output_dir, workspace)
    from bench_args import configure_system_certificates
    configure_system_certificates()
    from aicomputing import VirtualRuntime
    from aicomputing.backends.openai_compat import OpenAICompatBackend
    from aicomputing.backends.base import detect_tool_use
    from aicomputing.request import ExecuteRequest
    from result_schema import RESULT_COLUMNS, build_row, new_run_id
    key = read_key(args.key_file)
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "runner_protocol": RUNNER_PROTOCOL, "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "answer_format": ANSWER_FORMAT,
        "runner_provenance": {"fresh_run_only": True, "resuming_prior_run": False,
                              "maintained_entry_point": "evaluation.runner"},
        "status": "running", "started_at_utc": utc_now(), "model": MODEL, "provider": "openai", "api_base": API_BASE,
        "generation": generation, "reasoning_requested": {"reasoning_effort": "none"},
        "tool_policy": {"tool_choice_omitted": True, "effective_tool_choice": "none(default_no_tools)",
                        "tools_offered": False, "execute_tools": False, "reject_returned_tool_indicators": True},
        "manifest_sha256": hashlib.sha256(prepared.manifest_bytes).hexdigest(), "runner_sha256": sha256(__file__),
        "source_sha256": actual_sources,
        "environment": prepared.environment,
        "dataset_release_sha256": manifest["dataset_release"]["sha256"],
        "requirements_sha256": manifest["reproducibility"]["requirements_sha256"],
        "budget": {"operational_usd": str(budget), "user_ceiling_usd": str(ceiling),
                   **pricing_snapshot(),
                   "reservation_is_not_a_request_token_limit": True,
                   "missing_cache_counts": "missing reads assumed zero; missing writes charged at write rate for all remaining input; explicit upper bound"},
        "sdk_retries": 0, "validation_retries": 0, "timeout_retries": TIMEOUT_RETRIES,
        "http_phase_timeout_seconds": timeout_seconds,
        "http_timeout_policy": {"connect_seconds": timeout_seconds, "read_seconds": timeout_seconds,
                                "write_seconds": timeout_seconds, "pool_seconds": timeout_seconds,
                                "sdk_timeout_seconds": timeout_seconds,
                                "timeout_is_total_request_deadline": False,
                                "uniform_for_all_cases_and_strategies": True,
                                "exception_diagnostics": "class identities and httpx phase only; no exception messages or request details"},
        "infrastructure_policy": {"timeout_without_response": "retry failed request at most twice; full reservation retained per attempt",
                                  "other_transport_without_response": "excluded without retry; full reservation retained",
                                  "partial_contract": "preserve completed prediction; retry only failed checker request",
                                  "parsing_failure": "scored without retry"},
        "service_tier_requested": "default",
        "http_follow_redirects": False, "cache": "disabled", "planned_evaluations": planned_evaluations,
        "cases_per_benchmark": cases_per_benchmark, "max_generation_attempts": max_attempts,
        "dataset_mode": "offline cached; selected content fingerprints verified", "source_files_modified": False,
        "reasoning_fallback": "none; explicit unsupported none stops this run before any separate low condition",
    }
    (output_dir / "runner.executed.py").write_bytes(Path(__file__).read_bytes())
    prepared.snapshot(output_dir)
    (output_dir / "prompting.executed.py").write_bytes((workspace / "aicomputing/src/aicomputing/prompting.py").read_bytes())
    for relative in actual_sources:
        target = output_dir / "executed-source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source_bytes = (workspace / relative).read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != actual_sources[relative]:
            raise ValueError("Source changed after preflight")
        target.write_bytes(source_bytes)
    rows = []
    request_stream = (output_dir / "requests.jsonl").open("x", encoding="utf-8", newline="\n")
    evaluation_stream = (output_dir / "evaluations.jsonl").open("x", encoding="utf-8", newline="\n")
    csv_stream = (output_dir / "results.csv").open("x", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=RESULT_COLUMNS)
    writer.writeheader()
    ledger = Ledger(request_stream, budget, ceiling, max_attempts, timeout_seconds=timeout_seconds)

    class WireAuditedBackend(OpenAICompatBackend):
        def cache_state(self):
            return None

        def _get_client(self):
            if self._client is None:
                sdk = build_sdk_client(self._api_key, timeout_seconds)
                self._sdk = sdk
                backend = self

                def create(**kwargs):
                    if (any(field in kwargs for field in FORBIDDEN_TOKEN_FIELDS)
                            or any(field in kwargs for field in ("tools", "functions", "function_call"))
                            or kwargs.get("reasoning_effort") != "none" or kwargs.get("model") != MODEL
                            or kwargs.get("temperature") != 0.0 or kwargs.get("top_p") != 1.0
                            or "tool_choice" in kwargs or "service_tier" in kwargs):
                        raise StopRun("sdk_wire_settings_guard")
                    kwargs["service_tier"] = "default"
                    client_id = kwargs.get("extra_headers", {}).get("X-Client-Request-Id")
                    if not isinstance(client_id, str) or not re.fullmatch(r"[0-9a-f-]{36}", client_id):
                        raise StopRun("missing_client_request_id")
                    backend._record["client_request_id"] = client_id
                    backend._record.update(sdk_wire_checked=True, sdk_token_limit_keys=[],
                                           sdk_reasoning_effort="none", sdk_tool_choice_omitted=True,
                                           effective_tool_choice="none(default_no_tools)", sdk_tools_offered=False,
                                           sdk_service_tier="default",
                                           sdk_dispatched=True)
                    append_json(request_stream, {"event": "sdk_dispatch", "attempt": backend._record["attempt"],
                                                 "client_request_id": client_id,
                                                 "started_at_utc": utc_now(),
                                                 "dispatched_messages": kwargs["messages"],
                                                 "http_phase_timeout_seconds": timeout_seconds,
                                                 "sdk_token_limit_keys": [], "reasoning_effort": "none",
                                                 "tool_choice_omitted": True, "effective_tool_choice": "none(default_no_tools)",
                                                 "tools_offered": False,
                                                 "service_tier": "default",
                                                 "temperature": 0.0, "top_p": 1.0})
                    sdk_response = sdk.chat.completions.create(**kwargs)
                    backend._sdk_response = safe_sdk_response(sdk_response, detect_tool_use)
                    from aicomputing.backends.diagnostics import safe_request_id
                    backend._sdk_response["provider_request_id"] = safe_request_id(getattr(sdk_response, "_request_id", None))
                    append_json(request_stream, {"event": "sdk_response", "attempt": backend._record["attempt"],
                                                 **backend._sdk_response})
                    return sdk_response

                self._client = SimpleNamespace(base_url=sdk.base_url,
                                               chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            return self._client

        def generate_chat(self, messages, **options):
            return generate_audited_request(self, messages, options, ledger=ledger,
                                            stream=request_stream, call=super().generate_chat)

        def close(self):
            if getattr(self, "_sdk", None) is not None:
                self._sdk.close()

    def checkpoint():
        metadata.update(completed_evaluations=len(rows), generation_attempts=ledger.attempts,
                        returned_responses=ledger.responses, wall_seconds=time.perf_counter() - started,
                        usage_based_charge_usd=str(ledger.usage_based_charge),
                        exact_reported_charge_usd=str(ledger.exact_reported_charge),
                        conservative_charge_usd=str(ledger.charged),
                        unknown_usage_reservations_usd=str(ledger.unknown_reservations))
        cells = []
        for benchmark in BENCHMARKS:
            for strategy in STRATEGIES:
                selected = [row for row in rows if row["benchmark"] == benchmark and row["strategy"] == strategy]
                calls = [record for record in ledger.records if record.get("benchmark") == benchmark and record.get("strategy") == strategy]
                cells.append({"benchmark": benchmark, "strategy": strategy, "completed": len(selected),
                              "scorable": sum(bool(row["scorable"]) for row in selected),
                              "correct": sum(bool(row["correct"]) for row in selected),
                              "validation_failures": sum(row["validation_status"] != "valid" for row in selected),
                              "generation_attempts": len(calls),
                              "timeout_retries": sum(record["request_attempt"] > 1 for record in calls),
                              "charge_usd": str(sum((Decimal(record["charge_usd"]) for record in calls), Decimal(0)))})
        summary = {name: metadata[name] for name in ("status", "completed_evaluations", "generation_attempts",
                   "returned_responses", "wall_seconds", "usage_based_charge_usd", "exact_reported_charge_usd",
                   "conservative_charge_usd", "unknown_usage_reservations_usd")}
        summary.update(stop_reason=metadata.get("stop_reason") or ledger.stop_reason, cells=cells,
                       timeout_retries=sum(record["request_attempt"] > 1 for record in ledger.records),
                       infrastructure_excluded_evaluations=sum(not row["scorable"] for row in rows),
                       wire_checks_passed=sum(record["sdk_wire_checked"] for record in ledger.records),
                       requests_with_missing_usage=sum(not record["usage_complete"] for record in ledger.records),
                       tool_responses=sum(bool(record.get("tool_use_detected")) for record in ledger.records),
                       truncated_requests=sum(record.get("finish_reason") == "length" for record in ledger.records),
                       input_tokens_reported=sum(record.get("input_tokens") or 0 for record in ledger.records),
                       output_tokens_reported=sum(record.get("output_tokens") or 0 for record in ledger.records),
                       cached_input_tokens_reported=sum(record.get("cached_input_tokens_reported") or 0 for record in ledger.records),
                       cached_input_tokens_accounted=sum(record.get("cached_input_tokens_accounted") or 0 for record in ledger.records),
                       cache_write_tokens_reported=sum(record.get("cache_write_tokens_reported") or 0 for record in ledger.records),
                       cache_write_tokens_accounted=sum(record.get("cache_write_tokens_accounted") or 0 for record in ledger.records),
                       responses_missing_cached_input_count=sum(record.get("cached_input_tokens_reported") is None for record in ledger.records if record.get("returned_model")),
                       responses_missing_cache_write_count=sum(record.get("cache_write_tokens_reported") is None for record in ledger.records if record.get("returned_model")),
                       requests_with_cache_cost_upper_bound=sum(bool(record.get("charge_is_upper_bound")) for record in ledger.records),
                       reasoning_tokens_reported=sum(record.get("reasoning_tokens") or 0 for record in ledger.records),
                       responses_missing_reasoning_usage=sum(record.get("reasoning_tokens") is None for record in ledger.records if record.get("returned_model")),
                       returned_models=sorted({record["returned_model"] for record in ledger.records if isinstance(record.get("returned_model"), str)}))
        atomic_json(output_dir / "metadata.json", metadata)
        atomic_json(output_dir / "summary.json", summary)

    checkpoint()
    try:
        for benchmark in BENCHMARKS:
            adapter = get_adapter(benchmark)
            for strategy in STRATEGIES:
                backend = WireAuditedBackend(model=MODEL, base_url=API_BASE, api_key=key, reasoning_effort="none",
                                             timeout_seconds=timeout_seconds, sdk_max_retries=0, request_diagnostics=True)
                runtime = VirtualRuntime(backend=backend, model=MODEL, strategy=strategy, validation_mode="strict")
                run_id = new_run_id(MODEL, benchmark, strategy)
                try:
                    for item in resolved[benchmark]:
                        ledger.context = {"benchmark": benchmark, "strategy": strategy, "item_id": item.item_id, "category": item.category}
                        if ledger.stop_reason:
                            raise StopRun(ledger.stop_reason)
                        before_count = len(ledger.records)
                        try:
                            result = runtime.execute(ExecuteRequest(signature=item.signature, docstring=item.docstring,
                                                     inputs=item.inputs, return_type=item.return_type, strategy=strategy,
                                                     answer_format=ANSWER_FORMAT,
                                                     model=MODEL, validation_mode="strict", **generation))
                        except RecoverableInfrastructure:
                            calls = ledger.records[before_count:]
                            identity = (benchmark, strategy, item.item_id)
                            if (not calls or any((call["benchmark"], call["strategy"], call["item_id"]) != identity for call in calls)
                                    or calls[-1]["attempt"] <= before_count):
                                raise StopRun("transport_failure_context_guard") from None
                            result = infrastructure_result(calls, model=MODEL, strategy=strategy)
                            append_json(request_stream, {"event": "evaluation_infrastructure_exclusion",
                                                         "attempts": [call["attempt"] for call in calls],
                                                         **ledger.context, "failure_type": result.failure_type,
                                                         "scorable": False, "retry": False,
                                                         "partial_reported_input_tokens": sum(call.get("input_tokens") or 0 for call in calls),
                                                         "partial_reported_output_tokens": sum(call.get("output_tokens") or 0 for call in calls),
                                                         "complete_usage": False})
                        calls = ledger.records[before_count:]
                        attempt_accounting = account_attempts(result, calls)
                        if ledger.stop_reason:
                            raise StopRun(ledger.stop_reason)
                        correct = (result.scorable
                                   and result.validation_status not in ("unparsed", "invalid_type", "invalid_constraint")
                                   and adapter.score(result.result, item))
                        row = build_row(
                            result, run_id=run_id, benchmark=benchmark,
                            item_id=item.item_id, signature=item.signature,
                            docstring=item.docstring, inputs=item.inputs,
                            expected=item.expected, correct=bool(correct),
                            invocation_pool=item.invocation_pool,
                            return_type_source=item.return_type_source,
                            generation=generation,
                        )
                        row.update(attempt_accounting)
                        writer.writerow(row)
                        csv_stream.flush()
                        os.fsync(csv_stream.fileno())
                        append_json(evaluation_stream, {"row": row})
                        rows.append(row)
                        checkpoint()
                        print(f"[{len(rows):3}/{planned_evaluations}] {benchmark}/{strategy} {item.item_id}: valid={row['validation_status']} correct={row['correct']} calls={ledger.attempts} cost=${ledger.charged:.5f}", flush=True)
                finally:
                    backend.close()
        metadata["status"] = "completed"
    except StopRun as exc:
        metadata.update(status="stopped_by_guard", stop_reason=ledger.stop_reason or str(exc), incomplete_context=ledger.context)
    except KeyboardInterrupt:
        metadata.update(status="interrupted", stop_reason="keyboard_interrupt", incomplete_context=ledger.context)
    except BaseException as exc:
        # Never allow an SDK exception string/body to enter logs or a traceback.
        metadata.update(status="failed", stop_reason=type(exc).__name__, incomplete_context=ledger.context)
    finally:
        metadata["finished_at_utc"] = utc_now()
        checkpoint()
        request_stream.close()
        evaluation_stream.close()
        csv_stream.close()
    print(f"Experiment {metadata['status']}: {len(rows)} evaluations, {ledger.attempts} calls, ${ledger.charged:.6f}.", flush=True)
    return 0 if metadata["status"] == "completed" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, InvalidOperation) as exc:
        # A preflight/credential error is safely named without exposing contents.
        print(f"Experiment preparation failed: {type(exc).__name__}.", file=sys.stderr)
        raise SystemExit(2) from None
