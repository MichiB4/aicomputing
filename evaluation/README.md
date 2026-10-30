# Reproducible evaluation workflow

This research tooling is separate from the installable oracle package. The
current runner supports GPT-6 Luna, HumanEval+, MBPP+, GSM8K, and the Direct,
CoT, Pseudocode and Contract strategies.

## 1. Install and obtain the reviewed data

From the repository root:

```bash
python -m pip install "./aicomputing[openai,benchmarks]"
```

Use the approved `dataset-release.json` supplied for the study. Preparation
copies its exact bytes; it does not download raw datasets, change task text,
repair targets, or use model responses for selection.

For new MBPP+ data preparation, the [published-target migration](../docs/evaluation.md#mbpp)
traces existing reviewed inputs to the dataset's literal output lists. It creates
a separate release and preserves all earlier releases/results. Inference needs no reference-execution cache.

## 2. Prepare a new experiment, offline

```bash
python -m evaluation.prepare --dataset-release "<path-to-approved-dataset-release.json>" --experiment-id luna-smoke --cases-per-benchmark 5 --out-dir manifests/prepared/luna-smoke --budget-usd 1.00 --user-ceiling-usd 1.00
```

The new directory contains `manifest.json`, `dataset-release.json`, and
`requirements.txt`. Existing directories are rejected. The manifest freezes
source hashes, Python and runtime dependency versions, model settings, selected
inputs, grading fingerprints, timeout/retry policy, price assumptions and budget.

The default seed is 42. Preparation first selects 100 cases per benchmark;
the five-case smoke test uses the first five of that same selection. For a full
study, repeat preparation with `--cases-per-benchmark 100`, a new experiment
ID and a new output directory. Reuse the seed and reviewed release to preserve
the selections. These public development cases are not an unseen holdout.

Five cases produce 60 evaluations and 75 nominal model calls; 100 cases produce
1,200 evaluations and 1,500 nominal calls. Timeout retries can increase those
figures to 225 and 4,500 attempts respectively, subject to the frozen budget.
No inference occurs during preparation.

## 3. Verify before inference

```bash
python -m evaluation.runner --manifest manifests/prepared/luna-smoke/manifest.json --preflight-only
```

This checks software versions, source hashes, input hashes, selected content and
the frozen execution policy. It needs no credential and creates no run directory.
To restore the recorded environment, use the Python version in the manifest and:

```bash
python -m pip install -r manifests/prepared/luna-smoke/requirements.txt
```

Platform information is recorded separately. Matching versions and settings
does not guarantee identical hosted model outputs. Deliberate software changes
require a new preparation; never rewrite an old experiment to pass a check.

## 4. Run into a new directory

Supply `OPENAI_API_KEY` through the environment or an ignored local key file:

```bash
python -m evaluation.runner --manifest manifests/prepared/luna-smoke/manifest.json --output-dir runs/luna-smoke-001 --key-file .env.local
```

Each request retains the 60-second HTTP phase timeout and up to two timeout
retries. Parsing and validation never trigger retries. No application output
token cap is sent, no tools are offered, and native reasoning effort is `none`.
The timeout is not a total wall-clock deadline. Price estimates retain the
reviewed assumptions dated 2026-10-26; they are not a fresh billing quotation.

A run snapshots the exact validated manifest, dataset, requirements, runner,
maintained source files and project metadata. Its metadata records the actual
software environment. Results, request logs and summaries stay in that new
directory. Reruns use another new directory; no automatic resume or overwrite
is supported. CLI budget overrides must equal the frozen manifest values.

The per-result CSV and JSONL rows use a compact 35-column schema. Task identity,
specifications, inputs, expected and predicted values, outcome diagnostics,
token totals, generation settings, prompts, and reasoning/transport records
remain in each row. The consolidated fields are:

| Field | Values and meaning |
|---|---|
| `structural_validity` | `valid`, `invalid`, or `not_assessed`; checks a public return annotation without converting the prediction |
| `contract_status` | `not_checked`, `unchanged`, or `revised`; records whether a completed check changed the answer, without claiming it is correct |
| `retries` | Validation retries plus timeout retries for this evaluation; excludes the removed token-budget retry detail |

`benchmark_item_id` identifies the selected task. Its invocation ID, selection
seed, dataset revision, and other run-level provenance remain in the frozen
manifest. Per-request logs retain physical attempt counts and partial reported
usage for cost auditing. If usage is incomplete, the row's token totals remain
unknown (blank in CSV), never zero-filled. Pre-revision values and scores are
omitted from new result rows and are not separately scored by the runner.
Historical files retain their original columns; the new format requires a
freshly prepared, source-pinned manifest.

## 5. Audit the completed run, offline

```bash
python -B -m evaluation.audit --run-dir runs/luna-smoke-001
```

The command prints a JSON report and exits with code 0 for PASS or 1 for FAIL.
It reads the run directory without changing it, importing archived code,
restoring the old SDK environment, or making model calls. It supports completed
runs from the current workflow (runner schema 7).

The audit checks frozen snapshots, selected-case coverage, CSV/JSONL agreement,
request and retry evidence, reported usage, price calculations, budget
reservations and summary totals. Unknown usage remains unknown and retains
its reservation. Infrastructure exclusions stay outside the accuracy
denominator; accuracy is null when no evaluation is scorable.

PASS means the archived records agree. It does not independently re-parse or
re-grade model answers, establish dataset semantics, or verify a provider invoice.
A failure identifies the first inconsistent check and the stages already passed.
The command never repairs records or changes a score.

## Publication and historical records

`manifests/prepared/`, `runs/`, `.publish/`, `exports/`, legacy result
directories and `tests/` are ignored by Git. This workflow publishes source
and instructions by default. To reproduce a study, also supply its prepared
input bundle and the matching source revision through the chosen archive;
a source-only checkout does not contain the reviewed data.

Saved evaluation archives remain local under `.publish/`. Retired root-level
inputs/configurations are archived separately from the active project. The
maintained runner does not read old runner scripts or caches.
