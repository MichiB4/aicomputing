# Benchmark and scoring reference

The evaluation tooling measures model predictions for concrete function inputs.
Run it with the [single-command workflow](../evaluation/README.md). Ordinary
oracle-package use does not require datasets or reference answers.

## Benchmarks and approaches

| Benchmark | Default task count | Approaches |
| --- | ---: | --- |
| HumanEval+ | 100 | Direct, CoT, Pseudocode, Contract, code generation/execution |
| MBPP+ | 100 | Direct, CoT, Pseudocode, Contract, code generation/execution |
| GSM8K | 100 | Direct, CoT, Pseudocode, Contract, code generation/execution |

All approaches, including code generation/execution, run through the package's
`VirtualRuntime` strategies. The evaluator handles selection, scoring, and
recording; the package owns prompting and the standard Python worker.

Select any subset and sample size using `--benchmarks`, `--strategies`, and
`--samples`. The sample size is per benchmark. Each task contributes one input,
shared across the selected approaches. Matching seeds and reviewed data produce
matching task/input selections across models. Smaller samples are prefixes of
larger samples, taken from the same seeded shuffle.

The default configuration produces 1,500 evaluations and 1,800 nominal model
calls. Contract uses a prediction call and a checking call. Timeout retries can
add calls. Native reasoning effort is configured independently of the approach.

The four prompting approaches predict the function's return value from its
specification and input. The code approach generates an implementation, executes
it in a standard Python worker, and grades its returned value against the same target.
GSM8K code generation writes arithmetic for the supplied word problem.

## Shared reviewed data

The reviewed dataset is included at `data/dataset-release.json`, with its release
identity, checksum and upstream notices in [the data guide](../data/README.md).
Another reviewed
release can be supplied with `--dataset-release`. Selection uses approved tasks
and observations, verifies their content, and records their fingerprints and
scoring profiles in the result JSON. Resume uses the saved tasks rather than
reselecting them or requiring the source dataset file.

Public specifications and examples are retained. Actual inputs exclude displayed
public-example inputs. Expected answers are grading-only data and are not passed
to the model. The evaluated reference implementation is not supplied to the
model or used to repair its output.

For MBPP+, reviewed targets originate from aligned published input/output lists.
The bundled release contains the frozen targets and provenance. Evaluation does
not run model-generated programs to establish expected answers.

These public benchmark cases are development data, not an unseen holdout.
Published examples and possible training overlap limit conclusions about generalization.

## Results and scoring

The evaluator owns benchmark outcome classification and accuracy eligibility.
Package results contain execution diagnostics; the evaluator adds `correct`,
`scorable`, and benchmark-specific failure labels to its CSV/JSON records.
The package's `execution_failure` is mapped from worker diagnostics to the
existing `code_*` labels; worker harness failures remain infrastructure failures.

Evaluation requests the package's `validation_mode="none"` to preserve parsed
outputs without type conversion. Successful parsing or execution is recorded as
`validation_status="not_checked"`; benchmark structural validity and correctness
are assessed separately. Ordinary package calls default to strict output checks.

Predictions use the shared `framed_answer_v1` protocol. An isolated malformed
result field such as `{"result": 7?}` is recovered as the unchanged text `"7?"`;
it is not repaired to the number 7. Empty or ambiguous fields and missing answer
boundaries are parse failures. Every raw response is retained.

Each task's saved scoring profile defines its comparator. HumanEval+ uses
structured comparison; MBPP+ uses its reviewed numeric, exact, or set-comparison
rule; GSM8K uses numeric comparison. Type validation is separate from semantic
correctness: a structurally valid answer may still be wrong.

Wrong answers, malformed outputs, and generated-code syntax, policy, runtime, or
execution-timeout failures count as incorrect. Exhausted API timeouts and provider
failures are reported as infrastructure exclusions. Capability accuracy divides
correct answers by scorable evaluations. The summary also reports correct answers
divided by all completed evaluations, including infrastructure failures.

No parsing, validation, or code-repair call is made automatically. The generated
code prompt requests a general syntax and semantic review before submission.
An explicit token cap is never expanded automatically after truncation.

Results are saved as one detailed CSV and one JSON. They retain prompts, raw
responses, generated code, execution evidence, settings, usage, and previous
attempts. Explicit retries replace the current row with the latest outcome while
preserving earlier history and charges. Completed evaluations are skipped when
resuming an interrupted run.

Costs distinguish reported usage from conservative reservations for calls with
unknown usage. Model profiles record price assumptions; custom models supply
their rates explicitly.
