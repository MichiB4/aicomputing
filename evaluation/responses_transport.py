"""Text-only Responses transport with explicit native evidence and usage mapping."""
from __future__ import annotations

from types import SimpleNamespace


def normalize_data(data):
    """Retain visible output/usage; never request or retain encrypted reasoning."""
    output, text, unsupported = [], [], []
    for item in data.get("output") or []:
        kind = item.get("type")
        if kind == "reasoning":
            output.append({"type": "reasoning"})
        elif kind == "message":
            content = []
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    content.append({"type": "output_text", "text": part["text"]})
                    text.append(part["text"])
                elif part.get("type") == "refusal":
                    content.append({"type": "refusal", "refusal": part.get("refusal", "")})
                    text.append(part.get("refusal", ""))
                else:
                    unsupported.append(str(part.get("type")))
            output.append({"type": "message", "content": content})
        else:
            unsupported.append(str(kind))
    usage = data.get("usage") or {}
    incomplete = data.get("incomplete_details") or None
    finish = "length" if incomplete and incomplete.get("reason") == "max_output_tokens" else "stop"
    content = "\n".join(text)
    evidence = {"id": data.get("id"), "model": data.get("model"), "status": data.get("status"),
                "service_tier": data.get("service_tier"), "usage": usage,
                "incomplete_details": incomplete, "output": output, "unsupported_output_types": unsupported}
    raw = {"id": data.get("id"), "model": data.get("model"), "service_tier": data.get("service_tier"),
           "choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content}}],
           "usage": {"prompt_tokens": usage.get("input_tokens"), "completion_tokens": usage.get("output_tokens"),
                     "total_tokens": usage.get("total_tokens"),
                     "prompt_tokens_details": usage.get("input_tokens_details"),
                     "completion_tokens_details": usage.get("output_tokens_details")}}
    if unsupported:
        raw["tool_calls"] = unsupported
    return raw, evidence


def completion(response):
    raw, evidence = normalize_data(response.model_dump())
    choice = raw["choices"][0]
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=choice["finish_reason"],
        message=SimpleNamespace(content=choice["message"]["content"], tool_calls=raw.get("tool_calls"), function_call=None))],
        usage=SimpleNamespace(prompt_tokens=raw["usage"]["prompt_tokens"], completion_tokens=raw["usage"]["completion_tokens"]),
        model_dump=lambda: raw, _request_id=getattr(response, "_request_id", None), _native_response=evidence)
