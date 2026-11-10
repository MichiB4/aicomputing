"""Shared serialization, credentials, HTTP clients and token-cost accounting."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from aicomputing.answer_protocol import ANSWER_FORMAT

API_BASE = "https://api.openai.com/v1"
HTTP_PHASE_TIMEOUT_SECONDS = 180.0
SOURCE_ROOTS = ("aicomputing/src", "evaluation")

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

def replace_file(source, target):
    """Atomically replace a file, briefly retrying permission conflicts."""
    for attempt in range(5):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * 2 ** attempt)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    replace_file(temporary, path)

def read_key(path, key_env):
    """Read the credential named key_env without printing values or dotenv contents.

    key_env None means local inference, which takes no credential.
    """
    if key_env is None:
        if path is not None:
            raise ValueError("Local inference must not receive a credential file")
        return "ollama"
    if path is None:
        key = os.environ.get(key_env, "").strip()
        if not key:
            raise ValueError(key_env + " is absent; provide --key-file or the environment variable")
        return key
    found = []
    for original in path.read_text(encoding="utf-8-sig").splitlines():
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if separator and name.strip() == key_env:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if value and not any(character.isspace() for character in value):
                found.append(value)
    if len(found) != 1:
        raise ValueError("Credential file must contain exactly one nonempty " + key_env + " assignment")
    return found[0]

class StopRun(Exception):
    pass

def validate_timeout(seconds):
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("HTTP phase timeout must be a finite positive number of seconds")
    return float(seconds)

def build_sdk_client(api_key, timeout_seconds, *, base_url=API_BASE):
    """Configure both timeout layers without performing an HTTP request."""
    import httpx
    from openai import OpenAI
    timeout = httpx.Timeout(validate_timeout(timeout_seconds))
    http_client = httpx.Client(follow_redirects=False, timeout=timeout)
    try:
        return OpenAI(api_key=api_key, base_url=base_url, max_retries=0,
                      http_client=http_client, timeout=timeout)
    except BaseException:
        http_client.close()
        raise

def infrastructure_result(records, *, model, strategy):
    """Preserve incomplete call evidence without inventing a final answer."""
    from aicomputing.request import ExecuteResult, FailureType
    from aicomputing.status import ValidationStatus
    if not records or any(not record["sdk_dispatched"] for record in records):
        raise ValueError("Infrastructure outcome requires dispatched evidence")
    failure = FailureType.TIMEOUT if records[-1].get("exception_type") == "BackendTimeoutError" else FailureType.PROVIDER_ERROR
    prompts = [message["content"] for record in records for message in record["dispatched_messages"]
               if message.get("role") == "user"]
    responses = [record.get("response_text") for record in records if record.get("returned_model")]
    return ExecuteResult(result=None, raw_text="\n".join(text for text in responses if isinstance(text, str)),
                         model=model, strategy=strategy, failure_type=failure,
                         answer_format=ANSWER_FORMAT,
                         transport_diagnostics=[record.get("transport_diagnostics") for record in records],
                         validation_status=ValidationStatus.UNAVAILABLE, parse_status=None, prompts=prompts,
                         input_tokens=None, output_tokens=None, total_tokens=None,
                         model_calls=len(records), retries=0,
                         latency_ms=1000 * sum(record["elapsed_seconds"] for record in records),
                         reasoning_requests=[{"reasoning_effort": record.get("sdk_reasoning_effort", "none")} for record in records],
                         reasoning_observations=[({"reported_reasoning_tokens": record.get("reasoning_tokens"),
                                                  "returned_reasoning_chars": record.get("returned_reasoning_chars"),
                                                  "source": "provider_response"} if record.get("returned_model") else None)
                                                 for record in records])

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
    result.latency_ms = 1000 * sum(record["elapsed_seconds"] for record in records)
    result.reasoning_requests = [{"reasoning_effort": record.get("sdk_reasoning_effort", "none")} for record in records]
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
    result = {
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
    native = getattr(response, "_native_response", None)
    if native is not None:
        result["native_response"] = native
    return result

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
                    and bool(re.search(r"(?:unsupported value:\s*['\"]none['\"]|does not support\s*['\"]none['\"]|['\"]none['\"]\s+is not supported)", text))
                    and any(word in text for word in ("unsupported", "not supported", "invalid", "supported values"))
                )
        cause = getattr(cause, "__cause__", None)
        if cause is None:
            break
    return status, unsupported_none, identifiers

class Ledger:
    """Reserve, charge and record budgeted API attempts through ``emit``.

    Charges and attempt numbers continue from ``existing`` request records.
    """

    def __init__(self, emit, budget, profile, timeout_seconds=HTTP_PHASE_TIMEOUT_SECONDS,
                 *, max_attempts=None, existing=()):
        self.emit, self.budget, self.profile = emit, budget, profile
        self.model = profile.model
        self.max_attempts = max_attempts
        self.timeout_seconds = validate_timeout(timeout_seconds)
        self.attempts = max((record["attempt"] for record in existing), default=0)
        self.charged = sum((Decimal(record.get("charge_usd", "0")) for record in existing), Decimal(0))
        self.context, self.records = {}, []
        self.stop_reason = None

    def before(self, messages):
        if self.stop_reason or (self.max_attempts is not None and self.attempts >= self.max_attempts):
            self.stop_reason = self.stop_reason or "generation_attempt_cap"
            raise StopRun(self.stop_reason)
        input_bound = len(json.dumps(messages, ensure_ascii=True).encode("utf-8")) + self.profile.framing_reserve
        if input_bound > self.profile.max_input_reservation:
            self.stop_reason = "long_context_financial_guard"
            raise StopRun(self.stop_reason)
        # Cache writes cost more than ordinary input. Reserve the higher rate.
        reservation = (input_bound * self.profile.cache_write_rate
                       + self.profile.output_reservation * self.profile.output_rate)
        if self.charged + reservation > self.budget:
            self.stop_reason = "budget_reservation_guard"
            raise StopRun(self.stop_reason)
        self.attempts += 1
        record = {
            "attempt": self.attempts, **self.context, "started_at_utc": utc_now(),
            "requested_model": self.model, "application_max_new_tokens": None,
            "provider_output_reservation_tokens": self.profile.output_reservation,
            "input_reservation_tokens": input_bound, "reservation_usd": str(reservation),
            "sdk_wire_checked": False, "sdk_dispatched": False,
            "dispatched_messages": messages, "prompt_provenance": "captured before dispatch",
            "http_phase_timeout_seconds": self.timeout_seconds,
        }
        self.emit({"event": "request_start", **record})
        return record, reservation, time.perf_counter()

    def after(self, record, reservation, started, *, sdk_response=None, response=None, exception=None):
        record.update(elapsed_seconds=time.perf_counter() - started, finished_at_utc=utc_now(),
                      exception_type=type(exception).__name__ if exception else None)
        if sdk_response is not None:
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
            charge = ((input_tokens - effective_cached - effective_write) * self.profile.input_rate
                      + effective_cached * self.profile.cached_input_rate
                      + effective_write * self.profile.cache_write_rate
                      + output_tokens * self.profile.output_rate)
            record.update(cached_input_tokens_accounted=effective_cached,
                          cache_write_tokens_accounted=effective_write,
                          cached_input_missing_assumed_zero=cached_missing,
                          cache_write_missing_assumed_all_uncached_write=write_missing,
                          charge_is_upper_bound=cached_missing or write_missing,
                          charge_basis="reported_usage_cache_upper_bound" if cached_missing or write_missing else "reported_usage")
            if (input_tokens > record["input_reservation_tokens"]
                    or output_tokens > self.profile.output_reservation or charge > reservation):
                self.stop_reason = "provider_usage_exceeded_reservation"
        else:
            charge = reservation if record["sdk_dispatched"] else Decimal(0)
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
        native = record.get("native_response")
        if native is not None and native.get("status") != "completed":
            self.stop_reason = "truncated_code_or_text_response" if record.get("finish_reason") == "length" else "response_incomplete"
        if sdk_response is not None and record.get("returned_model") != self.model:
            self.stop_reason = "returned_model_mismatch"
        if record.get("finish_reason") == "length" or record.get("normalized_finish_reason") == "length":
            self.stop_reason = "provider_output_truncated"
        if not standard_tier:
            self.stop_reason = "service_tier_mismatch"
        if exception is not None:
            from aicomputing.backends.diagnostics import exception_diagnostics
            record["transport_diagnostics"] = getattr(exception, "diagnostics", None)
            record.update(exception_diagnostics(exception))
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
        self.emit({"event": "request_complete", **record})

def configure_system_certificates() -> None:
    """Configure optional system certificates during application startup."""
    try:
        import truststore
    except ModuleNotFoundError as exc:
        if exc.name != "truststore":
            raise
        return
    truststore.inject_into_ssl()
