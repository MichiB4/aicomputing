"""Allowlisted transport evidence, excluding exception text and request data."""

from __future__ import annotations

import re


def safe_request_id(value):
    """Accept only bounded opaque request identifiers, never arbitrary headers."""
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", value, flags=re.ASCII):
        return value
    return None


def exception_diagnostics(exception):
    """Inspect explicit causes; do not guess the cause of a provider delay."""
    import httpx

    types, phases, seen = [], [], set()
    status = request_id = None
    cause = exception
    for _ in range(12):
        if cause is None or id(cause) in seen:
            break
        seen.add(id(cause))
        qualified = type(cause).__module__ + "." + type(cause).__name__
        types.append(qualified if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,255}", qualified, flags=re.ASCII)
                     else "unrecorded_exception_class")
        for cls, phase in ((httpx.ConnectTimeout, "connect"), (httpx.ReadTimeout, "read"),
                           (httpx.WriteTimeout, "write"), (httpx.PoolTimeout, "pool")):
            if isinstance(cause, cls) and phase not in phases:
                phases.append(phase)
        candidate = getattr(cause, "status_code", None)
        if status is None and type(candidate) is int and 100 <= candidate <= 599:
            status = candidate
        if request_id is None:
            request_id = safe_request_id(getattr(cause, "request_id", None))
        cause = cause.__cause__
    return {"exception_cause_chain_types": types, "httpx_timeout_phases": phases,
            "timeout_phase_source": "httpx_exception_class" if phases else "unavailable",
            "http_status": status, "provider_request_id": request_id}
