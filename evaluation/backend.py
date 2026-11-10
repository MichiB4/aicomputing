"""Recorded, budgeted API calls for the evaluation common."""
from __future__ import annotations

from decimal import Decimal
import time
import uuid

from evaluation import common
from evaluation.responses_transport import completion


class InfrastructureFailure(Exception):
    def __init__(self, kind="provider_error", *, fatal=False):
        self.kind, self.fatal = kind, fatal
        super().__init__(kind)


class ModelOutputFailure(Exception):
    pass


def make_backend(profile, settings, key, emit, existing):
    # Lazy package imports preserve python -m evaluation --help without optional SDKs.
    from aicomputing.backends.base import Backend, GenerateResponse, detect_tool_use, normalize_finish_reason
    from aicomputing.backends._text_utils import strip_thinking
    from aicomputing.backends.diagnostics import safe_request_id
    from openai import APIError, APITimeoutError, APIConnectionError

    class RecordedBackend(Backend):
        name = "evaluation"
        default_model = profile.model

        def __init__(self):
            self.ledger = common.Ledger(emit, Decimal(str(settings["budget"])), profile,
                                        settings["timeout"], existing=existing)
            self.client = None

        def close(self):
            if self.client is not None:
                self.client.close()
                self.client = None

        def generate(self, prompt, **options):
            return self.generate_chat([{"role": "user", "content": prompt}], **options)

        def generate_chat(self, messages, *, model=None, temperature=0.0, top_p=1.0,
                          max_new_tokens=None, stop=None, **unused):
            if unused or stop or model not in (None, profile.model):
                raise ValueError("Unsupported generation override")
            if max_new_tokens != settings["max_output_tokens"]:
                raise ValueError("An implicit token-limit change was requested")
            if self.client is None:
                self.client = common.build_sdk_client(key, settings["timeout"], base_url=profile.api_base)
            ledger = self.ledger
            first = None
            for attempt in range(settings["retries"] + 1):
                record, reservation, started = ledger.before(messages)
                first = first or record["attempt"]
                body = {"model": profile.model, "messages": messages, "reasoning_effort": settings["reasoning"]}
                if profile.api_format == "responses":
                    body = {"model": profile.model, "input": messages,
                            "reasoning": {"effort": settings["reasoning"]}, "store": False}
                    if max_new_tokens is not None:
                        body["max_output_tokens"] = max_new_tokens
                else:
                    body.update(temperature=temperature, top_p=top_p)
                    if max_new_tokens is not None:
                        body["max_tokens"] = max_new_tokens
                if profile.service_tier:
                    body["service_tier"] = profile.service_tier
                record.update(logical_request_first_attempt=first, request_attempt=attempt + 1,
                              application_max_new_tokens=max_new_tokens, sdk_wire_checked=True, sdk_dispatched=True,
                              sdk_reasoning_effort=settings["reasoning"], sdk_api_format=profile.api_format,
                              client_request_id=str(uuid.uuid4()), request_body=body)
                emit({"event": "sdk_dispatch", **record})
                try:
                    headers = {"X-Client-Request-Id": record["client_request_id"]}
                    response = (completion(self.client.responses.create(**body, extra_headers=headers))
                                if profile.api_format == "responses" else self.client.chat.completions.create(**body, extra_headers=headers))
                except APIError as exc:
                    status = getattr(exc, "status_code", None)
                    retryable = (isinstance(exc, APIConnectionError)
                                 or status in (408, 409, 429)
                                 or (isinstance(status, int) and 500 <= status <= 599))
                    retry = retryable and attempt < settings["retries"]
                    delay = 2 ** min(attempt, 3) if retry else 0
                    record.update(retryable_error=retryable, retry_scheduled=retry,
                                  retry_delay_seconds=delay)
                    ledger.after(record, reservation, started, exception=exc)
                    ledger.stop_reason = None
                    if retry:
                        time.sleep(delay)
                        continue
                    kind = "timeout" if isinstance(exc, APITimeoutError) else "provider_error"
                    raise InfrastructureFailure(kind, fatal=not retryable) from None
                # Other exceptions propagate; the persisted dispatch keeps its unknown-usage reservation.
                safe = common.safe_sdk_response(response, detect_tool_use)
                safe["raw_response"] = safe.get("native_response", response.model_dump())
                safe["provider_request_id"] = safe_request_id(getattr(response, "_request_id", None))
                ledger.after(record, reservation, started, sdk_response=safe)
                issue = ledger.stop_reason
                ledger.stop_reason = None
                if safe.get("returned_model") != profile.model:
                    raise common.StopRun("returned_model_mismatch")
                if safe.get("tool_use_detected"):
                    raise ModelOutputFailure("tool_use_detected")
                if safe.get("finish_reason") == "length":
                    raise ModelOutputFailure("output_truncated")
                if safe.get("native_response", {}).get("status", "completed") != "completed":
                    raise ModelOutputFailure("response_incomplete")
                if issue:
                    raise common.StopRun(issue)
                raw = response.model_dump()
                choice = response.choices[0]
                text = choice.message.content or ""
                return GenerateResponse(text=text, model=profile.model,
                    text_for_parsing=strip_thinking(text),
                    input_tokens=safe["input_tokens"], output_tokens=safe["output_tokens"],
                    requested_reasoning={"reasoning_effort": settings["reasoning"]},
                    observed_reasoning={"reported_reasoning_tokens": safe.get("reasoning_tokens"),
                                        "returned_reasoning_chars": safe.get("returned_reasoning_chars"), "source": "provider_response"},
                    finish_reason=normalize_finish_reason(choice.finish_reason), provider_finish_reason=choice.finish_reason,
                    raw=raw, transport_diagnostics={"client_request_id": record["client_request_id"],
                                                  "elapsed_seconds": time.perf_counter() - started})
            raise AssertionError("Unreachable retry state")

    return RecordedBackend()
