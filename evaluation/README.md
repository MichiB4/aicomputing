# Run an evaluation with one command

First install this checkout in editable mode from the repository root:

```powershell
python -m pip install -e "./aicomputing[openai,benchmarks]"
```

The evaluator requires OpenAI SDK 2.30.0 or newer within major version 2.
The `benchmarks` extra optionally enables system certificates for HTTPS calls.
Evaluation checks that the imported package comes from this checkout so its
recorded source hashes describe the code actually used.

Then run:

```powershell
python -m evaluation --model gpt-6-luna --samples 100 --output results/luna-none
```

This runs HumanEval+, MBPP+, and GSM8K with Direct, CoT, Pseudocode, Contract,
and code generation followed by execution. It writes exactly two result files:

- `results/luna-none.csv`: one row per selected task/approach, including inputs,
  expected and predicted answers, correctness, errors, prompts, raw responses,
  generated code, execution details, timing, tokens, costs, and attempt/session history.
- `results/luna-none.json`: settings, provenance, the selected tasks and inputs,
  summaries, request details, attempt history, and progress for resuming the run.

The evaluator assigns correctness, benchmark failure labels, and `scorable`
eligibility from the returned values and execution diagnostics. These scoring
fields are retained in the CSV/JSON outputs.

Prediction-parsing diagnostics use `parse_status` and `parse_error`. Generated
code uses `execution_error` and execution details instead; its parsing fields
are empty. These diagnostics do not change benchmark failure labels or scoring.

Invalid Unicode in generated source is recorded as a code syntax failure; the
run continues. JSON and JSON-encoded CSV histories preserve the original text.
Standalone CSV text fields escape invalid Unicode characters as `\ud800`-style
sequences so the CSV remains valid UTF-8.

Input verification, selection, scoring, and checkpoints happen inside the command.

Concurrent commands targeting the same output are rejected using a persistent
lock file under the system temporary folder's `aicomputing-evaluation-locks`
directory. The result folder still contains only the CSV and JSON. A lock file's
presence does not mean a run is active: the operating system releases the lock
when the process closes it or exits.

The checkout includes the reviewed dataset at `data/dataset-release.json`; normal
evaluation needs no separate dataset download or preparation. Its release identity,
checksum and upstream notices are documented in [the data guide](../data/README.md).
For another reviewed release, supply `--dataset-release path/to/dataset-release.json`.
The command does not silently substitute raw data.
There are no per-run source/dataset copies, generated-program files, manifests,
or separate log directories.

## Choose benchmarks, approaches, and settings

All five approaches use the package's `VirtualRuntime` strategy interface,
including `code_execute`. Source generation and the standard Python worker live
in the package. Evaluation supplies benchmark data, selects `word_problem` mode
for GSM8K, and records and scores the returned evidence. The worker uses the
application's Python interpreter, installed packages and normal permissions.

Evaluation explicitly uses `validation_mode="none"` and zero validation retries.
The runtime preserves parsed outputs and reports `not_checked`; the evaluator
assesses types and correctness against the reviewed benchmark data.

Prediction strategies use one final-answer block. A clearly isolated malformed
result such as `{"result": 7?}` is preserved as the text `"7?"` and graded as
returned. Missing, empty, or ambiguous answers are parse failures. Full raw
responses are retained for both outcomes.

```powershell
python -m evaluation --model gpt-6-luna --reasoning max --benchmarks humaneval_plus mbpp_plus gsm8k --strategies direct cot pseudocode contract code_execute --samples 100 --timeout 900 --retries 1 --budget 20 --output results/luna-max
```

`--samples` is the number of tasks **per benchmark**. Any positive size up to the
approved pool size is accepted. Each task uses one input, shared across all selected
approaches. With the same seed and data, selections match across models; smaller
samples are prefixes of larger samples. The default seed is 42.

One hundred tasks across three benchmarks and five approaches give 1,500 evaluation
rows. Contract normally makes two model calls per evaluation, so this configuration
has 1,800 nominal API calls before retries for temporary API failures.

Frequently used options:

| Option | Default / meaning |
| --- | --- |
| `--model` | `gpt-6-luna`; Sol, Terra, Cerebras Qwen, and local Qwen profiles also work |
| `--benchmarks` | `humaneval_plus mbpp_plus gsm8k`; choose any subset |
| `--strategies` / `--approaches` | All five approaches; choose any subset |
| `--samples` | 100 per benchmark |
| `--reasoning` | `none`; native effort, independent of the prompting approach |
| `--temperature`, `--top-p` | 0 and 1 with reasoning off; omitted with native reasoning enabled |
| `--max-output-tokens` | No application cap; a supplied cap includes native reasoning tokens |
| `--timeout` | HTTP phase timeout: 180 seconds without reasoning, 900 with reasoning |
| `--retries` | One additional attempt on temporary API failures; no parse/validation/repair retries |
| `--execution-timeout`, `--execution-retries` | Code worker limits; inherit timeout/retries unless explicitly set |
| `--budget` | USD 20 conservative total guard, including previous attempts when resuming/retrying |
| `--output` | Required output basename; also accepts a `.csv` or `.json` suffix |

Temporary API failures include timeouts, connection errors, and HTTP 408, 409,
429 and 5xx responses. Retries wait 1, 2, 4, then at most 8 seconds between calls.
Every call retains its request history and cost accounting; failed calls with
unknown usage retain their conservative reservation. Request history also records
whether a retry was scheduled and its planned delay. If temporary failures exhaust
the retry limit, that evaluation is an infrastructure exclusion and the run
continues. Authentication and other permanent request errors stop the run.
The SDK's own retries remain disabled.

GPT-6.1 Sol requires `--reasoning low` or a higher effort; it does not support `none`.

Token-cap exhaustion is recorded as an invalid output; the runner never silently
increases the cap. Raw code is compiled and executed through the standard Python
worker. The source prompt requests syntax and semantic review before submission.

Use `OPENAI_API_KEY` or `CEREBRAS_API_KEY`, or provide a named assignment file with
`--key-file .env.local`. Local Ollama models require no key. Credentials are not
written into the results. The standard hosted profiles include recorded price
assumptions. For another model, specify `--provider`, `--input-price` and
`--output-price` in USD per million tokens, and either an application token cap or
`--provider-output-tokens`. `--base-url` supports a custom endpoint.

Preview selections and counts without inference or result files:

```powershell
python -m evaluation --model gpt-6-luna --samples 10 --output results/preview --dry-run
```

## Optional saved configuration

The example configuration contains the same settings as the flags:

```powershell
python -m evaluation --config evaluation/example-config.json
python -m evaluation --config evaluation/example-config.json --samples 100 --output results/full
```

Explicit flags override configuration values. Paths inside a configuration file
are relative to that file. Only the CSV and JSON are run outputs; the configuration
file is optional input.

## Resume or retry in the same two files

Ctrl+C checkpoints the current progress. Resume unfinished evaluations with:

```powershell
python -m evaluation --output results/luna-max --resume
```

Completed evaluations are skipped. An interrupted evaluation starts again, preserving
its earlier calls and any unknown-usage reservation. The JSON contains the selected
task data, so resuming does not require the original dataset file. It does require
the original package and evaluation source code, model settings, and task selection.
Both resume and retry compare the current source files and package configuration
against the original run's recorded hashes; replacement fingerprints cannot
override this check. Source changes require the original checkout or a new output
basename. Saved results remain readable. Operational timeouts, retry limits, and
the total budget may be adjusted explicitly.

JSON and CSV saves briefly retry permission conflicts during file replacement.
If saving the dispatch checkpoint fails, the request is not sent and receives no
usage reservation charge. Calls that may have reached the provider retain their
conservative reservation when usage is unknown. Failed sessions retain their
exception type and traceback in JSON and the CSV's `session_history` after resume;
exception messages are omitted because provider errors may contain credentials.

Retry only exhausted infrastructure failures (the default for `--retry-failed`):

```powershell
python -m evaluation --output results/luna-max --retry-failed --timeout 900 --retries 1
```

Or select failure labels from the CSV:

```powershell
python -m evaluation --output results/luna-max --retry-failed timeout parse_failure code_format_error
python -m evaluation --output results/luna-max --retry-failed all
```

Unknown failure labels are rejected before credentials are read or results are
changed. With no labels, `--retry-failed` selects API timeouts and provider errors.

The latest result becomes the current row, including if a repeated attempt is
incorrect. All previous outputs, generated programs, settings, and costs remain
in both files. A fresh run needs another output basename; existing results are
never overwritten accidentally.
If a retry is interrupted, the CSV retains the prior completed result together
with its matching generated code and execution evidence. The unfinished attempt
remains in the history.

Client setup or cleanup failures still finalize the session and save the
checkpoint. Cleanup diagnostics appear in JSON and the CSV's `session_history`;
completed evaluation results remain available, and the command returns a nonzero
exit status when cleanup fails.

Worker harness failures, such as failed process cleanup setup, save a completed
attempt with generated source, raw response, diagnostics, tokens and costs, then
stop the run with a nonzero exit status. They are excluded from capability
accuracy. After resolving the worker issue, retry the saved failure explicitly
with `--retry-failed harness_error`; the earlier attempt remains in the history.

Wrong answers, malformed outputs, and generated-code failures count as incorrect.
Exhausted API timeouts, provider failures and worker harness failures are excluded from capability accuracy
and reported separately. The JSON summary also reports correct answers divided
by all completed evaluations. Costs distinguish reported usage from conservative
reservations for calls whose usage was not returned.
