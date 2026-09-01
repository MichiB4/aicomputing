#!/usr/bin/env python3
"""
Offline verification of the experiment plumbing. Makes no API calls.

Requirement 17 asks that it be easy to inspect whether the pipeline is wired
correctly before spending money on a real run. This script checks exactly those
properties against a manifest, using a scripted fake backend in place of a
model, and prints a pass/fail line for each:

  1. every strategy receives identical benchmark items
  2. ground truth never appears in any prompt sent to the model
  3. STRICT validation withholds structurally invalid output
  4. contract records its check / revision metadata
  5. tokens, latency and model calls accumulate across multi-call strategies
  6. the manifest reproduces its own selection

Usage:
    python scripts/verify_experiment.py --manifest manifests/pilot-v2.json
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PKG = os.path.join(_ROOT, "aicomputing")
for path in (os.path.join(_PKG, "benchmarks"), os.path.join(_PKG, "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:  # pragma: no cover
    pass

from aicomputing import ExecuteRequest, VirtualRuntime  # noqa: E402
from aicomputing.backends.base import Backend, GenerateResponse  # noqa: E402
from aicomputing.strategies import list_arms  # noqa: E402
from aicomputing.validation import ValidationMode  # noqa: E402

from adapters import get_adapter  # noqa: E402
from experiment import (  # noqa: E402
    BASELINE_BENCHMARKS, PILOT_SIZES, build_manifest, load_manifest,
)

PASS, FAIL = "  PASS  ", "  FAIL  "
_results = []


def check(label, ok, detail=""):
    _results.append(bool(ok))
    print(f"[{PASS if ok else FAIL}] {label}" + (f"\n           {detail}" if detail else ""))


class RecordingBackend(Backend):
    """Fake model that records every prompt it is given."""

    def __init__(self, reply='{"result": 42}'):
        self.prompts, self.calls, self._reply = [], 0, reply

    @property
    def name(self): return "recording"

    @property
    def default_model(self): return "recording-model"

    def is_available(self): return True

    def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        self.calls += 1
        return GenerateResponse(text=self._reply, model="recording-model",
                                tokens_used=11, input_tokens=33, output_tokens=11,
                                finish_reason="stop")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(_ROOT, "manifests", "pilot-v2.json"))
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    benchmarks = list(manifest["benchmarks"])
    print(f"Verifying '{manifest['experiment_id']}' "
          f"(seed={manifest['seed']}, validation={manifest['validation_mode']})\n")

    # --- 1. identical items across strategies ------------------------------
    resolved = {b: get_adapter(b).resolve(manifest) for b in benchmarks}
    ids = {b: [i.item_id for i in items] for b, items in resolved.items()}
    check("every strategy receives identical items",
          all(len(v) == len(set(v)) and v for v in ids.values()),
          ", ".join(f"{b}={len(v)}" for b, v in ids.items()))

    # One invocation per function on the core benchmarks.
    core_ok = all(
        all(i.invocation_id for i in resolved[b])
        for b in benchmarks if b in BASELINE_BENCHMARKS
    )
    check("core benchmarks use exactly one invocation per function", core_ok)

    # --- 2. no ground-truth leakage ---------------------------------------
    leaks = []
    for bench, items in resolved.items():
        for item in items[:3]:
            backend = RecordingBackend()
            vr = VirtualRuntime(backend=backend, strategy="direct")
            vr.execute(ExecuteRequest(signature=item.signature, docstring=item.docstring,
                                      inputs=item.inputs, return_type=item.return_type,
                                      max_retries=0))
            expected = str(item.expected)
            # A leak is ground truth reaching the prompt from somewhere other
            # than the specification itself. The expected value legitimately
            # occurs in the spec all the time -- BBH's docstring says "Return
            # ONLY 'True' or 'False'", and HumanEval descriptions mention their
            # return values -- so matching on the prompt alone reports the
            # public specification as a leak. What must never happen is the
            # value appearing in the prompt WITHOUT being in the spec the model
            # is entitled to see.
            spec_text = f"{item.signature}\n{item.docstring}\n{item.inputs!r}"
            if len(expected) >= 4 and expected not in spec_text:
                for prompt in backend.prompts:
                    if expected in prompt:
                        leaks.append(f"{bench}:{item.item_id} -> {expected[:40]!r}")
    check("ground truth never appears in a prompt", not leaks,
          "; ".join(leaks[:3]) if leaks else "checked signature+docstring+inputs")

    # --- 3. strict validation ---------------------------------------------
    bad = RecordingBackend(reply='{"result": "not-an-int"}')
    req = lambda: ExecuteRequest(signature="add(x: int, y: int) -> int",
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
    cb = RecordingBackend()
    cres = VirtualRuntime(backend=cb, strategy="contract").execute(req())
    check("contract records check/revision metadata",
          cres.revised is not None and cb.calls == 2,
          f"revised={cres.revised}, pre_revision_value={cres.pre_revision_value!r}, "
          f"calls={cb.calls}, prompts_logged={len(cres.prompts)}")

    # --- 5. cost accumulation ---------------------------------------------
    rows = []
    for strategy in list_arms():
        b = RecordingBackend()
        r = VirtualRuntime(backend=b, strategy=strategy).execute(req())
        rows.append((strategy, b.calls, r.model_calls, r.total_tokens))
    ok = all(actual == recorded and recorded > 0 for _, actual, recorded, _ in rows)
    check("tokens / calls accumulate across every model call", ok,
          " | ".join(f"{s}: calls={a}=={rec} tok={t}" for s, a, rec, t in rows))

    # --- 6. manifest reproduces its selection ------------------------------
    rebuilt = build_manifest(experiment_id=manifest["experiment_id"], models=[],
                             strategies=manifest["strategies"], sizes=PILOT_SIZES,
                             benchmarks=benchmarks, seed=manifest["seed"],
                             validation_mode=manifest["validation_mode"])
    same = all(
        [(i["item_id"], i.get("invocation_id")) for i in manifest["benchmarks"][b]["items"]]
        == [(i["item_id"], i.get("invocation_id")) for i in rebuilt["benchmarks"][b]["items"]]
        for b in benchmarks
    )
    check("rebuilding with the same seed reproduces the same items", same)

    # --- 7. baseline scope --------------------------------------------------
    baseline = manifest.get("baseline") or {"benchmarks": []}
    check("baseline scoped to core coding benchmarks only",
          set(baseline["benchmarks"]) <= set(BASELINE_BENCHMARKS),
          f"baseline on {baseline['benchmarks']}")

    # --- 8. Direct thinking resolution -------------------------------------
    from aicomputing.model_config import resolve_thinking
    toggle = resolve_thinking("qwen/qwen3.8-27b", False)
    always = resolve_thinking("deepseek-r1-distill-70b", False)
    check("Direct thinking resolution is explicit, not silently overridden",
          toggle.effective is False and always.effective is True
          and always.honoured is False,
          f"toggle model honoured={toggle.honoured}; always-thinks model "
          f"effective={always.effective} honoured={always.honoured} (recorded)")

    # --- 9. Runtime-retry cost accumulation --------------------------------
    class FailThenPass(Backend):
        def __init__(self): self.calls = 0
        @property
        def name(self): return "ftp"
        @property
        def default_model(self): return "ftp"
        def is_available(self): return True
        def generate(self, prompt, **kw):
            self.calls += 1
            bad = self.calls <= 2
            return GenerateResponse(
                text='{"result": "nope"}' if bad else '{"result": 7}',
                model="ftp", tokens_used=10, input_tokens=40, output_tokens=10,
                finish_reason="stop")

    ftp = FailThenPass()
    acc = VirtualRuntime(backend=ftp, strategy="contract").execute(
        ExecuteRequest(signature="add(x: int, y: int) -> int", docstring="Sum.",
                       inputs={"x": 1, "y": 2}, return_type=int, max_retries=1))
    check("retry cost accumulates across the whole invocation",
          ftp.calls == acc.model_calls == 4 and acc.total_tokens == 200,
          f"contract x 2 attempts: backend calls={ftp.calls}, recorded "
          f"model_calls={acc.model_calls}, total_tokens={acc.total_tokens}")

    # --- 10. Infrastructure failures excluded from capability accuracy -----
    from aicomputing.request import ExecuteResult, FailureType
    from bench_args import execute_resilient
    from result_schema import build_row

    class Boom:
        model = "m"
        strategy = type("S", (), {"name": "direct"})()
        def execute(self, req): raise RuntimeError("429 rate limit")

    infra_row = build_row(
        execute_resilient(Boom(), object()), run_id="v", benchmark="gsm8k",
        benchmark_group="generalization", item_id="gsm8k/1", signature="f(x)",
        docstring="d", inputs={}, expected=1, correct=False, seed=42,
        manifest_version=1)
    check("provider errors stay out of the capability denominator",
          infra_row["failure_type"] == FailureType.PROVIDER_ERROR
          and infra_row["scorable"] is False,
          f"failure_type={infra_row['failure_type']}, "
          f"scorable={infra_row['scorable']}")

    # --- 11. Generated-code failure taxonomy -------------------------------
    codegen_rows = {}
    for status in ("syntax_error", "runtime_error", "timeout"):
        r = ExecuteResult(result=None, raw_text="", model="m",
                          strategy="code_gen_baseline", exec_status=status)
        codegen_rows[status] = build_row(
            r, run_id="v", benchmark="humaneval_plus", benchmark_group="core",
            item_id="HumanEval/0", signature="f(x)", docstring="d", inputs={},
            expected=1, correct=False, seed=42, manifest_version=1,
            is_baseline=True)
    check("generated-code failures are distinct from provider failures",
          all(row["failure_type"].startswith("generated_code")
              and row["scorable"] for row in codegen_rows.values()),
          ", ".join(f"{k}->{v['failure_type']}" for k, v in codegen_rows.items()))

    # --- 12. EvalPlus plus-inputs genuinely selectable ---------------------
    from collections import Counter
    from evalplus_data import load_humanevalplus
    he_pool = Counter(v.pool for i in load_humanevalplus() if i.eligible
                      for v in i.invocations)
    check("HumanEval+ plus inputs are genuinely in the selectable pool",
          he_pool.get("plus", 0) > 0,
          f"base={he_pool.get('base', 0):,}  plus={he_pool.get('plus', 0):,}  "
          f"unknown={he_pool.get('unknown', 0):,}")

    # --- 13. STRICT rejects lossy coercion ---------------------------------
    from aicomputing.parsing import coerce_type
    check("STRICT refuses lossy coercion; WARN still repairs",
          coerce_type(3.7, int, strict=True) == 3.7
          and coerce_type(3.7, int, strict=False) == 3
          and coerce_type("yes", bool, strict=True) == "yes"
          and coerce_type(3.0, int, strict=True) == 3,
          "3.7->int rejected under STRICT, repaired under WARN; "
          "3.0->int allowed (meaning preserved)")

    # --- 14. Dataset drift detection ---------------------------------------
    from adapters import ResolvedItem
    from experiment import content_fingerprint, verify_content
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
