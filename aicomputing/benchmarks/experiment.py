"""
Manifest-driven experiment configuration, item pools and reproducibility checks.

build_manifest() selects item IDs and invocation IDs once. The manifest-driven
runner resolves and evaluates those entries; legacy oracle scripts retain
separate sampling paths.

Results are reported per benchmark. benchmark_group distinguishes the core
specification-execution benchmarks from generalization benchmarks. The
code-generation comparison is restricted to the core benchmarks.

GSM8K and BBH use positional IDs, whose meaning depends on dataset contents and
ordering. check_pools() compares ID-pool fingerprints; it cannot detect every
content change. verify_content() compares fingerprints of selected resolved
specifications, inputs and expected outputs.

Manifest format and selection compatibility are versioned separately.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from aicomputing.model_config import get_provider_preset
from aicomputing.request import validate_generation_limits

from selection import (
    SELECTION_ALGORITHM_VERSION,
    derive_index,
    select_items,
    select_stratified,
)

# Version 2 separates prompting strategies from backend-native reasoning settings.
MANIFEST_VERSION = 2

# Default seed for manifest construction.
DEFAULT_SEED = 42

CORE_BENCHMARKS = ("humaneval_plus", "mbpp_plus")
GENERALIZATION_BENCHMARKS = ("gsm8k", "bbh")
ALL_BENCHMARKS = CORE_BENCHMARKS + GENERALIZATION_BENCHMARKS

BENCHMARK_GROUP: Dict[str, str] = {
    **{b: "core" for b in CORE_BENCHMARKS},
    **{b: "generalization" for b in GENERALIZATION_BENCHMARKS},
}

# This experiment includes the code-generation baseline for core benchmarks.
# The runner rejects baseline runs on the generalization benchmarks.
BASELINE_BENCHMARKS = frozenset(CORE_BENCHMARKS)
BASELINE_NAME = "code_gen_baseline"

# Five configured BBH task variants, each with a specification in
# bbh_oracle.TASK_SPECS. Adding a variant requires a matching specification.
# Report this subset as BBH-5; preserve individual task names in row categories.
BBH_DISPLAY_NAME = "BBH-5"
BBH_SUBSET_NOTE = (
    "five configured task variants from BIG-Bench Hard; not the full suite"
)

BBH_TASKS = (
    "date_understanding",
    "boolean_expressions",
    "tracking_shuffled_objects_three_objects",
    "logical_deduction_five_objects",
    "web_of_lies",
)

# Main experiment budgets. HumanEval+ includes every eligible function.
MAIN_SIZES: Dict[str, Optional[int]] = {
    "humaneval_plus": None,   # None = all eligible
    "mbpp_plus": 200,
    "gsm8k": 200,
    "bbh": 200,
}

# Pilot budgets for plumbing checks.
PILOT_SIZES: Dict[str, Optional[int]] = {
    "humaneval_plus": 5,
    "mbpp_plus": 5,
    "gsm8k": 10,
    "bbh": 10,               # up to two items per configured task
}

# ---------------------------------------------------------------------------
# Item pools
# ---------------------------------------------------------------------------

@dataclass
class PoolItem:
    """Candidate metadata before selection and adapter resolution.

    Text-benchmark loaders may leave signature and docstring empty until
    resolution. payload retains loader-specific data.
    """
    item_id: str
    signature: str
    docstring: str
    category: Optional[str] = None          # BBH task; None elsewhere
    invocation_ids: List[str] = field(default_factory=list)
    payload: Any = None                     # loader-specific extra data


def content_fingerprint(*, signature: str, docstring: str, inputs: Any,
                        expected: Any, category: Optional[str] = None) -> str:
    """Hash the specification, category, inputs and expected output."""
    payload = canonical_repr([signature, docstring, category, inputs, expected])
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def canonical_repr(value: Any) -> str:
    """Encode supported built-in values deterministically with type tags.

    Preserve dictionary order and distinguish container types. Sets are
    order-independent. Unsupported types and cyclic containers raise.
    """
    active: set[int] = set()

    def encode(item: Any) -> Any:
        kind = type(item)
        if item is None:
            return ["none"]
        if kind is bool:
            return ["bool", item]
        if kind is int:
            return ["int", hex(item)]
        if kind is float:
            return ["float", item.hex()]
        if kind is str:
            return ["str", item]
        if kind is bytes:
            return ["bytes", item.hex()]

        if kind not in (list, tuple, dict, set, frozenset):
            raise TypeError(
                "Unsupported fingerprint value type: "
                f"{kind.__module__}.{kind.__qualname__}"
            )

        identity = id(item)
        if identity in active:
            raise ValueError("Cyclic values cannot be fingerprinted")

        active.add(identity)
        try:
            if kind is dict:
                return [
                    "dict",
                    [[encode(key), encode(val)] for key, val in item.items()],
                ]

            entries = [encode(entry) for entry in item]
            if kind in (set, frozenset):
                entries.sort(key=json.dumps)
            return [kind.__name__, entries]
        finally:
            active.remove(identity)

    return json.dumps(encode(value), ensure_ascii=True, separators=(",", ":"))


def _fingerprint(values: Sequence[str]) -> str:
    """Hash an ordered ID sequence; callers sort when order is irrelevant."""
    payload = canonical_repr(list(values))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def dataset_revision(dataset: Any) -> Optional[str]:
    """Describe metadata exposed by an already loaded dataset.

    Return None when version/fingerprint attributes are unavailable. This is
    loader-reported metadata, not a verified source commit identifier.
    """
    version = getattr(getattr(dataset, "info", None), "version", None)
    fingerprint = getattr(dataset, "_fingerprint", None)
    parts = [str(value) for value in (version, fingerprint) if value]
    return "/".join(parts) or None


def load_pool(benchmark: str) -> List[PoolItem]:
    """Load candidate items, preserving the existing list-returning interface."""
    items, _ = _load_pool_with_revision(benchmark)
    return items


def _load_pool_with_revision(benchmark: str) -> tuple[List[PoolItem], Optional[str]]:
    """Load candidates and metadata together, without a second dataset lookup."""
    if benchmark in ("humaneval_plus", "mbpp_plus"):
        from evalplus_data import LOADERS
        items = LOADERS[benchmark]()
        pool = [
            PoolItem(
                item_id=item.item_id,
                signature=item.signature,
                docstring=item.docstring,
                invocation_ids=[invocation.invocation_id for invocation in item.invocations],
                payload=item,
            )
            for item in items if item.eligible
        ]
        # These loaders expose items, not the loaded dataset's metadata.
        # Selected-content fingerprints still cover resolved observations.
        return pool, None

    if benchmark == "gsm8k":
        from datasets import load_dataset
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        pool = [
            PoolItem(item_id=f"gsm8k/{index}", signature="", docstring="", payload=row)
            for index, row in enumerate(dataset)
        ]
        return pool, dataset_revision(dataset)

    if benchmark == "bbh":
        from datasets import load_dataset
        pool: List[PoolItem] = []
        revisions: Dict[str, Optional[str]] = {}
        for task in BBH_TASKS:
            dataset = load_dataset("lukaemon/bbh", task, split="test")
            revisions[task] = dataset_revision(dataset)
            for index, row in enumerate(dataset):
                pool.append(PoolItem(
                    item_id=f"{task}/{index}", signature="", docstring="",
                    category=task, payload=row))
        revision = (
            json.dumps(revisions, sort_keys=True)
            if any(value is not None for value in revisions.values()) else None
        )
        return pool, revision

    raise ValueError(f"unknown benchmark {benchmark!r}. Known: {ALL_BENCHMARKS}")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def resolve_generation_settings(
    generation: Any,
    model_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve experiment settings with zero temperature and valid counts."""
    if not isinstance(generation, dict):
        raise ValueError("generation must be an object")
    overrides = {} if model_config is None else model_config
    if not isinstance(overrides, dict):
        raise ValueError("model configuration must be an object")
    resolved = {
        "temperature": 0.0, "top_p": 1.0,
        "max_retries": 1, "max_new_tokens": None,
    }
    unknown = [str(key) for key in generation if key not in resolved]
    if unknown:
        raise ValueError("Unknown generation settings: " + ", ".join(unknown))
    # Validate the global block before applying per-model overrides.
    for layer in (generation, overrides):
        resolved.update({key: layer[key] for key in resolved if key in layer})
        temperature = resolved["temperature"]
        if type(temperature) not in (int, float) or temperature != 0.0:
            raise ValueError("Experiment temperature must be 0.0")
        top_p = resolved["top_p"]
        if type(top_p) not in (int, float) or not 0.0 <= top_p <= 1.0:
            raise ValueError("Experiment top_p must be numeric and between 0 and 1")
        validate_generation_limits(resolved["max_new_tokens"], resolved["max_retries"])
        resolved["temperature"] = 0.0
        resolved["top_p"] = float(top_p)
    return resolved


def resolve_model_connection(config: Any) -> Dict[str, Any]:
    """Resolve non-secret connection defaults for a frozen model entry."""
    if not isinstance(config, dict):
        raise ValueError("model configuration must be an object")
    for key in ("api_key", "apiKey", "token", "secret"):
        if key in config:
            raise ValueError(f"Credentials must not be stored in model configuration ({key})")

    provider = config.get("provider")
    backend = config.get("backend")
    base_url = config.get("base_url")
    for key, value in (("provider", provider), ("backend", backend), ("base_url", base_url)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be a nonempty string or null")
    if provider is None and backend is None:
        raise ValueError("model configuration requires provider or backend")
    if provider is not None:
        provider = provider.lower()
        preset = get_provider_preset(provider)
        if preset is None:
            raise ValueError("Unknown provider; use backend and base_url for a custom endpoint")
        if backend is None:
            backend = "ollama" if provider == "ollama" else "openai"
        if base_url is None:
            base_url = preset["base_url"]
    if backend == "hf" and (provider is not None or base_url is not None):
        raise ValueError("HF configuration uses backend/model, without provider or API URL")
    return {"provider": provider, "backend": backend, "base_url": base_url}


def resolve_reasoning_settings(
    backend: str,
    reasoning: Any,
    cli_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate native overrides and reject CLI disagreement with the manifest."""
    if not isinstance(backend, str) or not backend:
        raise ValueError("backend must be a nonempty string")
    if not isinstance(reasoning, dict):
        raise ValueError("reasoning must be an object; use {} for no override")

    allowed = {
        "openai": {"reasoning_effort": str},
        "ollama": {"think": (bool, str)},
        "hf": {"enable_thinking": bool},
    }.get(backend, {})
    for key, value in reasoning.items():
        if key not in allowed:
            raise ValueError(f"Unsupported reasoning setting {key!r} for {backend!r}")
        if not isinstance(value, allowed[key]):
            raise ValueError(f"Invalid type for reasoning setting {key!r}")
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"Reasoning setting {key!r} must not be empty")

    if cli_settings and cli_settings != reasoning:
        raise ValueError("CLI reasoning settings conflict with the manifest")
    return dict(reasoning)


def build_manifest(
    *,
    experiment_id: str,
    models: Sequence[str],
    strategies: Sequence[str],
    sizes: Dict[str, Optional[int]],
    benchmarks: Sequence[str] = ALL_BENCHMARKS,
    seed: int = DEFAULT_SEED,
    validation_mode: str = "strict",
    include_baseline: bool = True,
    generation: Optional[Dict[str, Any]] = None,
    model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    attach_fingerprints: bool = True,
) -> Dict[str, Any]:
    """
    Perform every selection and return the manifest as a plain dict.

    Reruns use the recorded selection. When attach_fingerprints is True,
    every selected item must resolve and receive a content fingerprint.
    Run-time verification compares that content; dataset metadata alone
    does not prove that the same observations are being evaluated.
    """
    generation_settings = resolve_generation_settings({} if generation is None else generation)
    configs = model_configs if model_configs is not None else {}
    if not isinstance(configs, dict):
        raise ValueError("model_configs must be an object keyed by model ID")
    frozen_configs = {}
    for model, config in configs.items():
        connection = resolve_model_connection(config)
        reasoning = resolve_reasoning_settings(
            connection["backend"], config.get("reasoning", {}),
        )
        effective_generation = resolve_generation_settings(generation_settings, config)
        generation_overrides = {
            key: effective_generation[key] for key in effective_generation if key in config
        }
        frozen_configs[model] = {
            **config, **connection, "reasoning": reasoning, **generation_overrides,
        }

    manifest: Dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "selection_algorithm_version": SELECTION_ALGORITHM_VERSION,
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "validation_mode": validation_mode,
        "models": list(models),
        "strategies": list(strategies),
        "baseline": {
            "name": BASELINE_NAME,
            "benchmarks": sorted(BASELINE_BENCHMARKS & set(benchmarks)),
        } if include_baseline else None,
        # Validated global settings; model entries override only fields they contain.
        "generation": generation_settings,
        # Per-model provider/backend settings and generation overrides.
        # Optional "reasoning" contains native backend overrides, shared by
        # every strategy for that model. Omitted or {} sends no override.
        # These are requested settings, not proof of the model's behavior.
        "model_configs": frozen_configs,
        "benchmarks": {},
    }

    for bench in benchmarks:
        pool, revision = _load_pool_with_revision(bench)
        by_id = {p.item_id: p for p in pool}
        if len(by_id) != len(pool):
            raise ValueError(f"{bench}: candidate pool contains duplicate item IDs")
        k = sizes.get(bench)

        if bench == "bbh":
            # Allocate the requested sample across the configured tasks.
            strata = {t: [p.item_id for p in pool if p.category == t] for t in BBH_TASKS}
            chosen = select_stratified(
                strata, k if k is not None else len(pool), seed, salt=bench,
            )
            ids = [i for t in sorted(chosen) for i in chosen[t]]
            stratum_counts = {t: len(v) for t, v in chosen.items()}
        else:
            all_ids = [p.item_id for p in pool]
            ids = select_items(all_ids, k if k is not None else len(all_ids),
                               seed, salt=bench)
            stratum_counts = None

        entries = []
        for item_id in ids:
            p = by_id[item_id]
            entry: Dict[str, Any] = {"item_id": item_id}
            if p.category:
                entry["category"] = p.category
            if p.invocation_ids:
                # Choose one invocation per function independently of sample size.
                idx = derive_index(seed, item_id, len(p.invocation_ids))
                entry["invocation_id"] = p.invocation_ids[idx]
            entries.append(entry)

        manifest["benchmarks"][bench] = {
            "group": BENCHMARK_GROUP[bench],
            "display_name": BBH_DISPLAY_NAME if bench == "bbh" else bench,
            "subset_note": BBH_SUBSET_NOTE if bench == "bbh" else None,
            "requested": k,
            "selected": len(entries),
            "pool_size": len(pool),
            "pool_fingerprint": _fingerprint(sorted(by_id)),
            "dataset_revision": revision,
            # Filled from resolved invocation sources during fingerprint attachment.
            "invocation_pool_policy": None,
            "strata": stratum_counts,
            "items": entries,
        }

    if attach_fingerprints:
        _attach_content_fingerprints(manifest)

    return manifest


def _attach_content_fingerprints(manifest: Dict[str, Any]) -> None:
    """Fingerprint every selected item using the runner's resolution adapters.

    Resolution or encoding failures abort construction rather than produce a
    manifest with incomplete content verification. Adapters reload sources;
    verification at run time checks whether that resolved content has changed.
    """
    from adapters import get_adapter  # Deferred: adapters imports this module.

    for benchmark, entry in manifest["benchmarks"].items():
        selected_ids = [item["item_id"] for item in entry["items"]]
        selected_set = set(selected_ids)
        if len(selected_set) != len(selected_ids):
            raise RuntimeError(f"{benchmark}: duplicate selected item IDs")

        resolved = {}
        for item in get_adapter(benchmark).resolve(manifest):
            if item.item_id in resolved:
                raise RuntimeError(f"{benchmark}: duplicate resolved item ID {item.item_id!r}")
            resolved[item.item_id] = item
        missing = [item_id for item_id in selected_ids if item_id not in resolved]
        unexpected = [item_id for item_id in resolved if item_id not in selected_set]
        if missing or unexpected:
            raise RuntimeError(
                f"{benchmark}: resolved items differ from the selection; "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )

        for item in entry["items"]:
            resolved_item = resolved[item["item_id"]]
            item["content_fingerprint"] = content_fingerprint(
                signature=resolved_item.signature, docstring=resolved_item.docstring,
                inputs=resolved_item.inputs, expected=resolved_item.expected,
                category=resolved_item.category,
            )
            if getattr(resolved_item, "invocation_pool", None):
                item["invocation_pool"] = resolved_item.invocation_pool
        entry["content_fingerprints"] = True
        entry["invocation_pool_policy"] = _describe_pool_policy(entry["items"])


def _describe_pool_policy(items: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Summarize selected invocation labels without inferring cache availability.

    No labels means source metadata is unavailable. When only some selected
    items have labels, include unknown rather than hide the missing metadata.
    """
    pools = {item.get("invocation_pool") for item in items if item.get("invocation_pool")}
    if not pools:
        return None
    if any(not item.get("invocation_pool") for item in items):
        pools.add("unknown")
    if pools == {"base"}:
        return "base_only"
    if pools == {"plus"}:
        return "plus_only"
    return "+".join(sorted(pools))


def verify_content(manifest: Dict[str, Any], benchmark: str, resolved,
                   *, hard: bool = True) -> List[str]:
    """Check content fingerprints and full selected-item coverage.

    Missing fingerprints, missing/unexpected items, duplicate IDs and changed
    content are verification failures. hard=False returns their diagnostics;
    hard=True rejects the run before model execution.
    """
    stored = {}
    problems: List[str] = []
    for entry in manifest_items(manifest, benchmark):
        item_id = entry["item_id"]
        if item_id in stored:
            problems.append(f"{benchmark}:{item_id} duplicate manifest item ID")
        stored[item_id] = entry.get("content_fingerprint")
        if not stored[item_id]:
            problems.append(f"{benchmark}:{item_id} missing content fingerprint")

    seen = set()
    for item in resolved:
        item_id = item.item_id
        if item_id in seen:
            problems.append(f"{benchmark}:{item_id} duplicate resolved item ID")
        seen.add(item_id)
        if item_id not in stored:
            problems.append(f"{benchmark}:{item_id} unexpected resolved item")
            continue
        wanted = stored[item_id]
        if not wanted:
            continue
        actual = content_fingerprint(
            signature=item.signature, docstring=item.docstring,
            inputs=item.inputs, expected=item.expected, category=item.category,
        )
        if actual != wanted:
            problems.append(
                f"{benchmark}:{item_id} content changed "
                f"({wanted[:12]} -> {actual[:12]})"
            )
    for item_id in stored:
        if item_id not in seen:
            problems.append(f"{benchmark}:{item_id} selected item was not resolved")

    if problems and hard:
        raise RuntimeError(
            f"{len(problems)} manifest content verification problem(s). "
            f"This run cannot reproduce the recorded selection and content. "
            f"Rebuild the manifest or restore its matching data. First: {problems[0]}"
        )
    return problems


def save_manifest(manifest: Dict[str, Any], path: str) -> str:
    """Replace a complete JSON manifest, preserving an existing target's mode.

    Resolve symbolic links so saving continues to update their target. New
    files retain the temporary file's private permissions.
    """
    target = os.path.realpath(path)
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    try:
        existing_mode = stat.S_IMODE(os.stat(target).st_mode)
    except FileNotFoundError:
        existing_mode = None

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory,
            prefix=".manifest-", suffix=".json", delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(manifest, handle, indent=2, sort_keys=False)
            handle.write("\n")
        if existing_mode is not None:
            os.chmod(temporary_path, existing_mode)
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except PermissionError:
                if os.name != "nt":
                    raise
                # Windows cannot delete a read-only temporary output.
                os.chmod(temporary_path, stat.S_IWRITE)
                os.unlink(temporary_path)
    return path


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    check_compatible(manifest)
    return manifest


def check_compatible(manifest: Dict[str, Any]) -> None:
    """Validate the root object and supported integer compatibility tags.

    These tags identify encoding/selection rules; they do not validate the
    full manifest schema, configuration or current dataset content.
    """
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    mv = manifest.get("manifest_version")
    if type(mv) is not int or mv != MANIFEST_VERSION:
        raise ValueError(
            f"manifest_version {mv} != {MANIFEST_VERSION} supported by this code. "
            f"Rebuild the manifest, or check out the revision that wrote it."
        )
    sv = manifest.get("selection_algorithm_version")
    if type(sv) is not int or sv != SELECTION_ALGORITHM_VERSION:
        raise ValueError(
            f"selection_algorithm_version {sv} != {SELECTION_ALGORITHM_VERSION}. "
            f"Selection or fingerprint rules differ, or the version is not an integer; "
            f"rebuild the manifest."
        )


def manifest_items(manifest: Dict[str, Any], benchmark: str) -> List[Dict[str, Any]]:
    entry = manifest["benchmarks"].get(benchmark)
    if entry is None:
        raise ValueError(
            f"benchmark {benchmark!r} is not in this manifest "
            f"(has: {sorted(manifest['benchmarks'])})"
        )
    return entry["items"]


def check_pools(manifest: Dict[str, Any], benchmark: str) -> Optional[str]:
    """Compare ID-pool fingerprints when the manifest records one.

    Return a warning on mismatch, otherwise None (matching or unchecked).
    Equal ID pools do not prove equal contents or row ordering. Selected
    resolved observations are checked separately by verify_content().
    """
    entry = manifest["benchmarks"].get(benchmark)
    recorded = entry.get("pool_fingerprint") if entry else None
    if not recorded:
        return None
    current = _fingerprint(sorted(p.item_id for p in load_pool(benchmark)))
    if current != recorded:
        return (f"{benchmark}: pool fingerprint changed "
                f"({recorded} -> {current}); item IDs may no longer refer to "
                f"the same examples")
    return None
