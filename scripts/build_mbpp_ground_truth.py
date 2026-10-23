#!/usr/bin/env python3
"""
Build the frozen MBPP+ ground-truth cache (offline preprocessing).

Why this exists
---------------
MBPP+ publishes 1,174 base and 39,841 plus inputs but NO expected outputs for
any of them. The only MBPP+ ground truth that exists as data is in `test_list`,
which covers base inputs alone. Without this step MBPP+ could only ever be
evaluated on its base inputs, and calling that "MBPP+" would overstate what was
measured.

What this does, and the boundary it respects
--------------------------------------------
It executes the official EvalPlus canonical solution on the published inputs to
compute expected outputs, ONCE, offline, and freezes the result.

That is benchmark construction, not AI computing. The distinction the thesis
rests on is about what the EVALUATED SYSTEM is given, not about what the
evaluator knows: an evaluator is always allowed to know the right answer. What
must never happen is the reference implementation reaching the system under
test.

Guarantees, enforced by construction:

  * This script is never imported by the runtime, the strategies, the
    prompting layer, or the code-generation baseline. Nothing under
    `src/aicomputing/` imports it, and `adapters.py` reads only the frozen
    artifact.
  * The artifact serialises ONLY item_id, invocation inputs, expected output,
    and provenance. `canonical_solution` is not written to it, so the
    implementation cannot leak downstream even by accident.
  * It runs before the experiment. The main run consumes the cache and never
    executes a canonical solution.

Isolation
---------
Each task's canonical solution runs in a separate process with a timeout, using
the same restricted-globals sandbox as the code-generation baseline. Inputs for
one task are batched into a single process because per-input spawning would be
~40,000 process starts on Windows.

This is not a security boundary against hostile code -- the inputs are a
published academic dataset -- but it does contain hangs and crashes, and a task
whose solution cannot be executed safely is marked ineligible rather than
having its isolation weakened.

Usage:
    python scripts/build_mbpp_ground_truth.py
    python scripts/build_mbpp_ground_truth.py --limit 20 --out data/mbpp_gt.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PKG = os.path.join(_ROOT, "aicomputing")
for path in (os.path.join(_PKG, "benchmarks"), os.path.join(_PKG, "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from bench_args import configure_system_certificates  # noqa: E402
from mbpp_ground_truth import (  # noqa: E402
    GROUND_TRUTH_VERSION, VALUE_ENCODING, decode_value, encode_value,
)

DEFAULT_OUT = os.path.join(_ROOT, "data", "mbpp_plus_ground_truth.json")


def _worker(code: str, entry_point: str, inputs, conn) -> None:
    """
    Child process: run the canonical solution over one task's inputs.

    Module-level so it survives pickling under the spawn start method.
    """
    try:
        from baselines.code_gen_baseline import _build_safe_globals
        namespace = {
            "__builtins__": _build_safe_globals(),
            "__name__": "__mbpp_canonical__",
        }
        exec(compile(code, "<mbpp_canonical>", "exec"), namespace)
        fn = namespace.get(entry_point)
        if fn is None:
            conn.send(("error", f"canonical solution defines no {entry_point!r}"))
            return
        out = []
        for args in inputs:
            try:
                frozen_inputs = encode_value(args)
                value = fn(*decode_value(frozen_inputs))
                # Snapshot before a later call can mutate returned containers.
                out.append(("ok", (frozen_inputs, encode_value(value))))
            except BaseException as exc:
                out.append(("error", f"{type(exc).__name__}: {exc}"))
        conn.send(("ok", out))
    except BaseException as exc:
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            conn.send(("error", "unreportable failure"))
    finally:
        conn.close()


def run_task(code: str, entry_point: str, inputs, timeout: float = 30.0):
    """Execute one task's canonical solution over its inputs, in isolation."""
    parent, child = multiprocessing.Pipe(duplex=False)
    proc = None
    try:
        proc = multiprocessing.Process(target=_worker,
                                       args=(code, entry_point, inputs, child))
        proc.start()
        child.close()
        try:
            if not parent.poll(timeout):
                proc.terminate()
                proc.join(2)
                return "timeout", None
            status, payload = parent.recv()
            return status, payload
        except EOFError:
            return "error", "worker died without reporting"
    finally:
        parent.close()
        child.close()
        if proc is not None and proc.pid is not None:
            if proc.is_alive():
                proc.terminate()
            proc.join(2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the MBPP+ ground-truth cache")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug: only process the first N tasks")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="Seconds per task (all of its inputs together)")
    args = ap.parse_args()
    if args.limit is not None and args.limit < 0:
        ap.error("--limit must be non-negative")
    # Windows reserves 0xffffffff milliseconds for an infinite wait.
    max_timeout = (2**32 - 2) / 1000
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= max_timeout:
        ap.error(f"--timeout must be finite and greater than 0, and at most {max_timeout:.3f} seconds")

    configure_system_certificates()

    from evalplus.data import get_mbpp_plus

    print("Loading MBPP+ from evalplus ...")
    tasks = get_mbpp_plus()
    task_ids = sorted(tasks)
    if args.limit is not None:
        task_ids = task_ids[: args.limit]

    entries = {}
    stats = {"tasks": len(task_ids), "tasks_ok": 0, "tasks_failed": 0,
             "base_inputs": 0, "plus_inputs": 0,
             "base_resolved": 0, "plus_resolved": 0, "input_failures": 0}
    failures = []

    for n, task_id in enumerate(task_ids, 1):
        task = tasks[task_id]
        base = [list(x) for x in (task.get("base_input") or [])]
        plus = [list(x) for x in (task.get("plus_input") or [])]
        stats["base_inputs"] += len(base)
        stats["plus_inputs"] += len(plus)
        all_inputs = base + plus
        if not all_inputs:
            continue

        status, payload = run_task(task["canonical_solution"],
                                   task["entry_point"], all_inputs, args.timeout)
        if status != "ok":
            stats["tasks_failed"] += 1
            failures.append({"task_id": task_id, "reason": f"{status}: {payload}"})
            print(f"[{n}/{len(task_ids)}] {task_id}: FAILED ({status})", flush=True)
            continue

        resolved = []
        for i, (ok, value) in enumerate(payload):
            pool = "base" if i < len(base) else "plus"
            if ok != "ok":
                stats["input_failures"] += 1
                continue
            resolved.append({"index": i, "pool": pool,
                             "inputs": value[0], "expected": value[1]})
            stats["base_resolved" if pool == "base" else "plus_resolved"] += 1

        if resolved:
            stats["tasks_ok"] += 1
            entries[task_id] = {"entry_point": task["entry_point"],
                                "invocations": resolved}
        else:
            stats["tasks_failed"] += 1
            failures.append({"task_id": task_id, "reason": "no input resolved"})

        if n % 50 == 0:
            print(f"[{n}/{len(task_ids)}] ...", flush=True)

    artifact = {
        "ground_truth_version": GROUND_TRUTH_VERSION,
        "value_encoding": VALUE_ENCODING,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "evalplus.data.get_mbpp_plus",
        "method": ("canonical solution executed offline in an isolated process "
                   "to compute expected outputs; the implementation itself is "
                   "NOT serialised and is never exposed to the runtime"),
        "stats": stats,
        "failures": failures,
        "tasks": entries,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=1, allow_nan=False)

    # The artifact must not contain the implementation under any key.
    blob = json.dumps(artifact)
    assert "canonical_solution" not in blob, "implementation leaked into the artifact"

    print("\n" + "=" * 58)
    for k, v in stats.items():
        print(f"   {k:16} {v}")
    print(f"   {'failures':16} {len(failures)}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
