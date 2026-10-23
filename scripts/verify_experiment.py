#!/usr/bin/env python3
"""Verify runtime plumbing with scripted responses, without model calls.

The default run uses synthetic items and does not load datasets. It checks
request/prompt plumbing, validation, contract metadata, usage accounting,
requested native settings, failure classification and content fingerprints.
These smoke checks do not establish model quality or complete leakage safety.

A supplied manifest is checked for compatibility and item-ID structure.
Resolving real items, rebuilding selections and checking the HumanEval+ pool
requires --check-datasets; that mode may download data, but never calls a model.

Usage:
    python scripts/verify_experiment.py
    python scripts/verify_experiment.py --manifest manifests/pilot-v3.json
    python scripts/verify_experiment.py --manifest manifests/pilot-v3.json --check-datasets
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PKG = os.path.join(_ROOT, "aicomputing")
for path in (os.path.join(_PKG, "benchmarks"), os.path.join(_PKG, "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from aicomputing import ExecuteRequest, VirtualRuntime  # noqa: E402
from aicomputing.backends.base import (
    Backend, BackendCommunicationError, BackendTimeoutError, GenerateResponse,
)  # noqa: E402
from aicomputing.parsing import coerce_type  # noqa: E402
from aicomputing.request import ExecuteResult, FailureType  # noqa: E402
from aicomputing.strategies import list_strategies  # noqa: E402
from aicomputing.validation import ValidationMode  # noqa: E402

from adapters import ResolvedItem, get_adapter  # noqa: E402
from experiment import (  # noqa: E402
    BASELINE_BENCHMARKS, MANIFEST_VERSION, build_manifest, content_fingerprint,
    load_manifest, resolve_reasoning_settings, verify_content,
)

from bench_args import configure_system_certificates, execute_resilient  # noqa: E402
from result_schema import build_row  # noqa: E402
from selection import select_items  # noqa: E402

PASS, FAIL = "  PASS  ", "  FAIL  "
_results = []


def check(label, ok, detail=""):
    _results.append(bool(ok))
    print(f"[{PASS if ok else FAIL}] {label}" + (f"\n           {detail}" if detail else ""))


class RecordingBackend(Backend):
    """Record prompts and return a constant or an explicit response sequence."""

    def __init__(self, reply='{"result": 42}', *, replies=None):
        self.prompts = []
        self.calls = 0
        self._reply = reply
        self._replies = None if replies is None else iter(replies)

    @property
    def name(self):
        return "recording"

    @property
    def default_model(self):
        return "recording-model"

    def is_available(self):
        return True

    def generate(self, prompt, **kwargs):
        text = self._reply if self._replies is None else next(self._replies)
        self.prompts.append(prompt)
        self.calls += 1
        return GenerateResponse(
            text=text, model=self.default_model, tokens_used=11,
            input_tokens=33, output_tokens=11, finish_reason="stop",
            requested_reasoning={},
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", help="Optional current-version manifest to inspect")
    ap.add_argument("--check-datasets", action="store_true",
                    help="Resolve/rebuild real datasets; may download data; requires --manifest")
    args = ap.parse_args()
    if args.check_datasets and not args.manifest:
        ap.error("--check-datasets requires --manifest")
    _results.clear()

    manifest = load_manifest(args.manifest) if args.manifest else None
    # The held-out sentinel tests field exclusion, not answer correctness.
    resolved = {"synthetic": [ResolvedItem(
        item_id="synthetic/1", signature="add(x: int, y: int) -> int",
        docstring="Return the sum.", inputs={"x": 1, "y": 2},
        expected="HELD_OUT_SENTINEL_8f21d6", return_type=int,
    )]}
    if args.check_datasets:
        configure_system_certificates()
        resolved = {b: get_adapter(b).resolve(manifest) for b in manifest["benchmarks"]}
        print(f"Dataset checks for '{manifest['experiment_id']}' (may load/download data)\n")
    else:
        print("Offline synthetic plumbing checks (no dataset loads or model calls)\n")

    # --- 1. Nonempty selections with unique item IDs -----------------------
    ids = ({b: [entry["item_id"] for entry in settings["items"]]
            for b, settings in manifest["benchmarks"].items()}
           if manifest is not None else
           {b: [item.item_id for item in items] for b, items in resolved.items()})
    valid_ids = {
        b: bool(v) and all(isinstance(item_id, str) and item_id.strip() for item_id in v)
        and len(v) == len(set(v))
        for b, v in ids.items()
    }
    check("selections contain nonempty, unique item IDs",
          bool(ids) and all(valid_ids.values()),
          ", ".join(f"{b}={len(v)}" for b, v in ids.items()))

    # One invocation per function on the core benchmarks.
    if manifest is not None:
        core_ok = all(
            valid_ids[b]
            and all(entry.get("invocation_id") for entry in settings["items"])
            for b, settings in manifest["benchmarks"].items() if b in BASELINE_BENCHMARKS
        )
        check("core manifest entries name one invocation per unique function", core_ok)

    # --- 2. Sampled held-out-value probe ---------------------------------------
    leaks = []
    for bench, items in resolved.items():
        for original in items[:3]:
            # Ordinary answers can occur in public instructions, e.g. True/False.
            # Replace only the probe copy's grading target with a distinct marker.
            spec_text = f"{original.signature}\n{original.docstring}\n{original.inputs!r}"
            sentinel = "HELD_OUT_SENTINEL_8f21d6"
            while sentinel in spec_text:
                sentinel += "_"
            item = replace(original, expected=sentinel)
            for strategy in list_strategies():
                backend = RecordingBackend()
                VirtualRuntime(backend=backend, strategy=strategy).execute(
                    ExecuteRequest(signature=item.signature, docstring=item.docstring,
                                   inputs=item.inputs, return_type=item.return_type,
                                   max_retries=0))
                if any(item.expected in prompt for prompt in backend.prompts):
                    leaks.append(f"{bench}:{item.item_id}:{strategy}")
    check("sampled prompts across registered strategies pass the held-out-value probe",
          not leaks, "; ".join(leaks[:3]) if leaks else
          "up to 3 items per benchmark; controlled sentinel smoke check, not a leakage proof")

    # --- 3. strict validation ---------------------------------------------
    bad = RecordingBackend(reply='{"result": "not-an-int"}')
    def req():
        return ExecuteRequest(signature="add(x: int, y: int) -> int",
                              docstring="Return the sum.", inputs={"x": 1, "y": 2},
                              return_type=int, max_retries=0)
    strict = VirtualRuntime(backend=bad, strategy="direct",
                            validation_mode=ValidationMode.STRICT).execute(req())
    warn = VirtualRuntime(backend=RecordingBackend(reply='{"result": "not-an-int"}'),
                          strategy="direct",
                          validation_mode=ValidationMode.WARN).execute(req())
    check("STRICT withholds invalid output, WARN preserves it",
          strict.result is None and warn.result == "not-an-int",
          f"strict={strict.result!r} ({strict.failure_type}), warn={warn.result!r}")

    # --- 4. contract metadata ---------------------------------------------
    for verdict, expected, revised in (
        ('{"verdict": "accept"}', 42, False),
        ('{"verdict": "revise", "result": 3}', 3, True),
        ('{"result": 99}', 42, False),
    ):
        cb = RecordingBackend(replies=['{"result": 42}', verdict])
        cres = VirtualRuntime(backend=cb, strategy="contract").execute(req())
        check(f"contract verdict {verdict} selects and records the intended answer",
              cres.result == expected and cres.revised is revised
              and cres.pre_revision_value == 42 and cb.calls == 2
              and len(cres.prompts) == 2,
              f"result={cres.result!r}, revised={cres.revised}, calls={cb.calls}")

    # --- 5. cost accumulation ---------------------------------------------
    rows = []
    for strategy in list_strategies():
        b = RecordingBackend()
        r = VirtualRuntime(backend=b, strategy=strategy).execute(req())
        rows.append((strategy, b.calls, r))
    ok = all(actual == r.model_calls and actual > 0
             and r.input_tokens == actual * 33 and r.output_tokens == actual * 11
             and r.total_tokens == actual * 44 and r.reasoning_requests == [{}] * actual
             for _, actual, r in rows)
    check("known usage, call counts and requested settings accumulate for every strategy", ok,
          " | ".join(f"{s}: calls={a}, recorded={r.model_calls}, tok={r.total_tokens}"
                     for s, a, r in rows))

    # --- 6. manifest reproduces its selection ------------------------------
    if args.check_datasets:
        rebuilt = build_manifest(
            experiment_id=manifest["experiment_id"], models=manifest["models"],
            strategies=manifest["strategies"],
            sizes={b: settings["requested"] for b, settings in manifest["benchmarks"].items()},
            benchmarks=list(manifest["benchmarks"]), seed=manifest["seed"],
            validation_mode=manifest["validation_mode"],
            generation=manifest["generation"], model_configs=manifest["model_configs"],
            include_baseline=bool(manifest.get("baseline")), attach_fingerprints=False,
        )
        same = all(
            [(i["item_id"], i.get("invocation_id")) for i in settings["items"]]
            == [(i["item_id"], i.get("invocation_id"))
                for i in rebuilt["benchmarks"][b]["items"]]
            for b, settings in manifest["benchmarks"].items()
        )
        check("recorded sizes and seed reproduce the manifest item/invocation selection", same)
        resolved_matches = all(
            [(item.item_id, item.invocation_id) for item in resolved[bench]]
            == [(entry["item_id"], entry.get("invocation_id")) for entry in settings["items"]]
            for bench, settings in manifest["benchmarks"].items()
        )
        check("adapters resolve every recorded item/invocation in manifest order", resolved_matches)
        for bench, items in resolved.items():
            missing = [entry["item_id"] for entry in manifest["benchmarks"][bench]["items"]
                       if not entry.get("content_fingerprint")]
            check(f"{bench}: recorded content fingerprints exist and match resolved items",
                  not missing and not verify_content(manifest, bench, items, hard=False))
    else:
        pool = [f"synthetic/{i}" for i in range(20)]
        selected = select_items(pool, 7, 42)
        check("synthetic selection is repeatable and input-order independent",
              len(selected) == len(set(selected)) == 7
              and selected == select_items(pool, 7, 42)
              and selected == select_items(list(reversed(pool)), 7, 42))
        print("[  SKIP  ] Real dataset selection, content and plus-pool checks; use --check-datasets")

    # --- 7. Baseline scope in a supplied manifest ---------------------------
    if manifest is not None:
        baseline = manifest.get("baseline") or {"benchmarks": []}
        check("baseline scoped to core coding benchmarks only",
              set(baseline["benchmarks"]) <= set(BASELINE_BENCHMARKS),
              f"baseline on {baseline['benchmarks']}")

    # --- 8. Explicit native configuration, without inferred capability -----
    configured = resolve_reasoning_settings("ollama", {"think": False})
    defaults = resolve_reasoning_settings("ollama", {})
    check("native reasoning overrides are explicit configuration",
          configured == {"think": False} and defaults == {},
          f"requested overrides={configured}; omitted overrides={defaults}; "
          "model compliance is not inferred")

    # --- 9. Runtime-retry cost accumulation --------------------------------
    ftp = RecordingBackend(replies=[
        '{"result": "nope"}', '{"verdict": "accept"}',
        '{"result": 7}', '{"verdict": "accept"}',
    ])
    acc = VirtualRuntime(backend=ftp, strategy="contract").execute(
        ExecuteRequest(signature="add(x: int, y: int) -> int", docstring="Sum.",
                       inputs={"x": 1, "y": 2}, return_type=int, max_retries=1))
    check("retry cost accumulates across the whole invocation",
          acc.result == 7 and ftp.calls == acc.model_calls == 4
          and acc.input_tokens == 132 and acc.output_tokens == 44
          and acc.total_tokens == 176 and acc.reasoning_requests == [{}, {}, {}, {}],
          f"contract x 2 attempts: backend calls={ftp.calls}, recorded "
          f"model_calls={acc.model_calls}, total_tokens={acc.total_tokens}")

    # --- 10. Infrastructure failures excluded from capability accuracy -----
    class Boom:
        model = "m"
        strategy = SimpleNamespace(name="direct")

        def __init__(self, error):
            self.error = error

        def execute(self, request):
            raise self.error

    for error, failure in (
        (BackendCommunicationError("429 rate limit"), FailureType.PROVIDER_ERROR),
        (BackendTimeoutError("request timed out"), FailureType.TIMEOUT),
        (RuntimeError("429 rate limit"), FailureType.HARNESS_ERROR),
    ):
        infra_row = build_row(
            execute_resilient(Boom(error), object()), run_id="v", benchmark="gsm8k",
            benchmark_group="generalization", item_id="gsm8k/1", signature="f(x)",
            docstring="d", inputs={}, expected=1, correct=False, seed=42,
            manifest_version=MANIFEST_VERSION)
        check(f"{type(error).__name__} is classified as {failure} and excluded",
              infra_row["failure_type"] == failure and infra_row["scorable"] is False)

    # --- 11. Generated-code failure taxonomy -------------------------------
    codegen_rows = {}
    for status in ("syntax_error", "runtime_error", "timeout"):
        r = ExecuteResult(result=None, raw_text="", model="m",
                          strategy="code_gen_baseline", exec_status=status)
        codegen_rows[status] = build_row(
            r, run_id="v", benchmark="humaneval_plus", benchmark_group="core",
            item_id="HumanEval/0", signature="f(x)", docstring="d", inputs={},
            expected=1, correct=False, seed=42, manifest_version=MANIFEST_VERSION,
            is_baseline=True)
    check("generated-code failures are distinct from provider failures",
          all(row["failure_type"].startswith("generated_code")
              and row["scorable"] for row in codegen_rows.values()),
          ", ".join(f"{k}->{v['failure_type']}" for k, v in codegen_rows.items()))

    # --- 12. EvalPlus plus-inputs genuinely selectable ---------------------
    if args.check_datasets and "humaneval_plus" in manifest["benchmarks"]:
        from evalplus_data import load_humanevalplus

        he_pool = Counter(v.pool for item in load_humanevalplus() if item.eligible
                          for v in item.invocations)
        check("HumanEval+ contains eligible plus-pool invocations",
              he_pool.get("plus", 0) > 0, str(dict(he_pool)))

    # --- 13. Both modes use the same limited coercion -----------------------
    check("WARN and STRICT share limited conversions without float truncation",
          all(coerce_type(3.7, int, strict=mode) == 3.7
              and coerce_type("yes", bool, strict=mode) == "yes"
              and type(coerce_type(3.0, int, strict=mode)) is int
              and coerce_type(3.0, int, strict=mode) == 3
              for mode in (False, True)),
          "3.7 and 'yes' stay unchanged; 3.0 becomes int 3; validation decides acceptance")

    # --- 14. Dataset drift detection ---------------------------------------
    probe = ResolvedItem(item_id="gsm8k/7", signature="solve(q: str) -> float",
                         docstring="Solve.", inputs={"q": "2+2?"}, expected=4.0)
    probe_manifest = {"benchmarks": {"gsm8k": {"items": [{
        "item_id": probe.item_id,
        "content_fingerprint": content_fingerprint(
            signature=probe.signature, docstring=probe.docstring,
            inputs=probe.inputs, expected=probe.expected)}]}}}
    moved = ResolvedItem(item_id="gsm8k/7", signature=probe.signature,
                         docstring=probe.docstring,
                         inputs={"q": "a different question"}, expected=4.0)
    same_ok = verify_content(probe_manifest, "gsm8k", [probe]) == []
    drift_caught = False
    try:
        verify_content(probe_manifest, "gsm8k", [moved])
    except RuntimeError:
        drift_caught = True
    check("dataset drift is detected by content, not position",
          same_ok and drift_caught,
          "same content verifies; the same ID pointing at a different "
          "problem is a hard error")

    failed = _results.count(False)
    print(f"\n{len(_results) - failed}/{len(_results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
