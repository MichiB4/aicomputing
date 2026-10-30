# Experimental evaluation

The research tooling is separate from the installable oracle package. Package
users need no benchmark datasets or reference answers; see the
[package guide](../aicomputing/README.md). For running a study, follow the
[reproducible workflow](../evaluation/README.md).

## Study design

| Benchmark | Selected tasks | Strategies |
|---|---:|---|
| HumanEval+ | 100 | Direct, CoT, pseudocode, contract |
| MBPP+ | 100 | Direct, CoT, pseudocode, contract |
| GSM8K | 100 | Direct, CoT, pseudocode, contract |

Each coding function contributes one concrete input. All strategies and models
receive the same selected tasks, specifications and inputs. This measures
return-value prediction from specifications and inputs, rather than the standard
code-generation pass@1 used by HumanEval and MBPP.

The full study has 1,200 evaluations per model and 1,500 nominal model calls:
contract uses a prediction call and a checking call. A five-task-per-benchmark
smoke check has 60 evaluations and 75 nominal calls. Timeouts can add attempts.
Confidence, self-consistency and hybrid remain available in the oracle package;
the four study strategies are a separately selected experimental condition.

These are public development pools, not an unseen holdout. Possible training
contamination and the limits of specification review must be reported with the
results.

## Reviewed data

| Source | Source tasks | Approved task pool | Quarantined |
|---|---:|---:|---:|
| HumanEval+ | 164 | 153 | 11 |
| MBPP+ | 378 | 343 | 35 |
| GSM8K test | 1,319 | 1,181 | 138 |

Review retains the original public descriptions and examples, clearly labels
examples, and selects a different actual input. Input exclusions cover repeated
examples, incompatible interfaces, unspecified domains and target disagreements.
Independent rules derived from public specifications and question arithmetic
were used for review, without evaluated model responses. This reduces preventable
data errors; it does not prove every natural-language specification unambiguous.

A release records every source task as approved or quarantined, its review
reason/evidence, typed inputs/targets, public-example exclusions and grading
profile. `prepared_data.py` validates the release before selection. Missing or
changed releases, unknown scoring profiles, conflicting duplicate targets and
insufficient approved pools stop preparation. Inference resolves only the frozen
release; it does not download datasets or execute reference implementations.

The current local release is
`manifests/prepared/mbpp-published-20261030-v2/dataset-release.json`.
Its SHA-256 is
`b961d334453779e42bd4e32df90b71fb2f177ff0be008a0dcc742ba273edff15`.
Prepared data is distributed separately from a source-only checkout.

### HumanEval+

[Dataset](https://huggingface.co/datasets/evalplus/humanevalplus).
The loader reads signature and docstring from the public prompt and extracts
aligned literal `inputs`/`results` lists from the test script. It never executes
that script. The reviewed release retains public examples and removes overlaps
with the selected actual input. Public helper context is included only where
explicitly reviewed; non-unique or insufficiently specified outputs are held.

### MBPP+

[Dataset](https://huggingface.co/datasets/evalplus/mbppplus/viewer/default/test).
The `test` column contains expanded module-level input/output lists. AST parsing
extracts literals without executing imports, test programs or reference solutions.
Python types and original pair indices (`published:N`) are preserved. The
reference code supplies only the function interface, never the model's prompt.

At cached revision `b2d74c91837c3f2a20c1299ae98133cbe7cfa077`, 375 tasks contain
precomputed lists. There are 39,917 supported pairs across 372 tasks; 376 pairs
contain unsupported complex/nonfinite values. `Mbpp/88`, `Mbpp/255` and `Mbpp/630`
compute targets through helpers instead of listing them. Raw loading can use
public base assertions for tasks without output lists. Malformed lists stop the
task; unsupported pairs are omitted without coercion or invented labels.

The current release maps 342 approved cases to exactly matching published values.
`Mbpp/630` retains its independently reviewed predecessor observation with explicit
provenance. All 343 frozen inputs, targets, pool labels and grading profiles are
preserved. The prior 35 quarantines remain; extraction alone approves no new
candidates. A reference-execution cache is neither loaded nor required.

Base labels match typed argument content in public assertions, including wrapped
calls. Unmatched calls use `unknown`, since public assertions do not form a
complete base catalog across revisions. Reviewed releases inherit their verified
base/plus labels by input identity and record that source.

To trace a reviewed release to a new local source snapshot, offline:

```bash
python -B scripts/prepare_mbpp_published.py --source-arrow PATH_TO_MBPP_ARROW --source-revision SOURCE_REVISION --previous-release PATH_TO_REVIEWED_RELEASE.json --release-id NEW_RELEASE_ID --out-dir manifests/prepared/NEW_RELEASE_DIRECTORY
```

The new directory contains the release, exact source Arrow bytes, predecessor
release, comparison report and pair-exclusion ledger. SHA-256 identifies the
actual bytes; the revision argument records supplied source metadata. Migration
checks public text, examples, interface and recognized scorer ASTs. Targets must
match exactly by typed input identity. Missing/conflicting observations are
quarantined without relabeling or resampling. Existing directories are rejected.

### GSM8K

[Dataset](https://huggingface.co/datasets/openai/gsm8k).
Each task provides a question and a worked solution ending in a `####` numeric
answer. Only the question is shown to the model. Preparation extracts the final
number and uses independent question arithmetic for review.

## Selection, output and scoring

The selection seed defaults to 42. Task IDs are sorted and sampled deterministically;
one reviewed input per coding function prevents functions with more generated
tests from receiving more statistical weight. The smoke check uses the prefix of
the same 100-task selection. Manifests pin selected input and content fingerprints.

The maintained runner uses `framed_text_v1`. Every strategy clearly separates
public examples, the actual task and the final-answer block. The parser reads
only that block. It does not use reference types to coerce predictions or repair
incomplete answers. An isolated malformed result can remain text with a parse
warning; a missing or ambiguous answer remains a parsing failure.

Grading uses the release's fixed profile: structured comparison for HumanEval+,
reviewed equality/tolerance or set equivalence for MBPP+, and numeric absolute
tolerance 0.001 for GSM8K. Reference values remain evaluation-only data.

Structural validity checks a public return annotation without converting the
prediction. Unannotated/Any returns are `not_assessed`; a single gold answer does
not establish a universal required type. Constraint validity checks only explicit
constraints. Semantic correctness is the frozen grader's comparison.

Incorrect answers distinguish parsing failure, declared structural/constraint
failure, reference-type mismatch and semantic wrong answer. Provider errors,
timeouts and harness errors are infrastructure exclusions outside capability
accuracy. Contract status records whether a check changed the prediction, not
whether its output is correct. Unknown token usage stays unknown.

## Future model runs

The [maintained workflow](../evaluation/README.md) prepares, preflights, executes
and audits GPT-6 Luna studies with source/environment snapshots, a frozen budget,
60-second HTTP phase timeouts and up to two timeout retries. Parsing/validation
failures do not trigger retries. No application output-token cap or tools are
supplied. The phase timeout is not a total wall-clock deadline.

Shared manifest-driven tools remain available for the planned local models:

```bash
python scripts/make_manifest.py --experiment-id local-study --dataset-release PATH_TO_RELEASE.json --model-configs PATH_TO_MODELS.json --out manifests/local-study.json
python scripts/run_experiment.py --manifest manifests/local-study.json --dry-run
python scripts/run_experiment.py --manifest manifests/local-study.json --out-dir runs/local-study
python scripts/aggregate_results.py --raw runs/local-study --out runs/local-study-summary.csv
```

Model configuration contains connection, native reasoning and generation settings,
never credentials. The generic launcher does not provide the Luna runner's audited
budget and timeout-retry machinery. Use the audited workflow for the paid Luna run.
Changes to software or experimental settings require a fresh prepared manifest.
Saved evaluation archives are outside the active workflow and remain local.
