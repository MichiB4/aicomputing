# Published evaluation results

[Download the verified result bundle](final-evaluations-2026-11-16.zip) (26.98 MiB).

Archive SHA256:

```text
d4ca74777afecdb2c635b1e574cb0726d91fe4318ab2e685ce9a4ea9bb4f9a54
```

Extract locally from the repository root:

```powershell
Expand-Archive results/final-evaluations-2026-11-16.zip -DestinationPath results/extracted-2026-11-16
```

On Linux/macOS, use `unzip` with the same archive and destination. Extracted
files and new evaluation outputs remain ignored by Git.

The bundle contains the four completed full benchmark experiments and the
latest completed TDMSci v2 paired case-study run. Each run retains its original
CSV and JSON bytes, settings, prompts, model replies, typed tasks, execution
evidence and recorded provenance.

## Benchmark experiments

Each experiment selects the same 100 tasks per benchmark from HumanEval+, MBPP+
and GSM8K, using seed 42. Each task is evaluated with Direct, CoT, Pseudocode,
Contract and code generation/execution: 1,500 evaluations per experiment.

| Model / reasoning | Correct / scorable | Capability accuracy | Correct / all completed |
| --- | ---: | ---: | ---: |
| Sol 6.1 low | 1,479/1,500 | 98.60% | 98.60% |
| Sol 6.1 max | 1,483/1,499 | 98.93% | 98.87% |
| Terra 6.1 max | 1,485/1,500 | 99.00% | 99.00% |
| Terra 6.1 none | 1,409/1,500 | 93.93% | 93.93% |

Sol-max has one correctly excluded infrastructure failure: HumanEval/136 with
Contract exhausted its API retries. Its capability denominator is 1,499; all
1,500 evaluations remain in the files. Output failures count as incorrect.

Reasoning `none` requests use temperature 0 and top-p 1. Native `low`/`max`
requests omit sampling overrides. No application output-token cap is set.
The saved settings record the operational timeouts and retry limits.

## Case study

The included run is `tdmsci-paired-100-terra-none-v2-20261116-repeat-2`, the latest
completed v2 run, selected chronologically. It uses the published TDMSci test
split, 100 sentences, seed 42, GPT-6.1 Terra, direct strategy and reasoning
`none`, with strict span validation and no automatic retries or runtime cache.

| Method | Precision | Recall | Micro F1 | Exact sentences |
| --- | ---: | ---: | ---: | ---: |
| Terra v2 | 57.94% | 75.61% | 65.61% | 40/100 |
| Training-phrase baseline | 46.15% | 43.90% | 45.00% | 29/100 |

One model output (`T0117`) fails the nonoverlapping-span contract. Its prediction
is withheld and the gold mentions count as misses. This is a completed model
attempt, with the original response and diagnostics retained.

## Contents and integrity

- `benchmarks/`: four CSV/JSON pairs (eight files).
- `case_study/`: the latest paired v2 CSV/JSON (two files).
- `manifest.json`: filenames, sizes, SHA256 hashes and run identities.
- `VERIFICATION.json`: the verification scope and outcome.
- `LICENSES/`: benchmark notices, TDMSci provenance and applicable code/data licenses.

All 6,000 benchmark evaluations were reparsed and independently rescored from
the saved model responses. Both the original and cleaned scoring pipelines
reproduce their rows and summaries. Re-executing the 884 distinct program/input
combinations reproduced all 1,200 saved typed worker outputs exactly. The case
study was independently rescored from its saved token spans and raw replies.
CSV/JSON fields and source/dataset hashes were checked; no discrepancies were
found. The audit sent no additional provider requests.

The JSON files preserve the dataset identity and source hashes recorded when
the runs were made. The bundled task selections and grading targets are
self-contained. Resuming or retrying a saved experiment requires matching its
original source hashes; a fresh run uses the current checkout.

## Interpretation

These are single-run measurements on public benchmark pools. They do not
establish performance on an unseen holdout. TDMSci contains duplicate sentences
and training/test overlap; the corpus provenance describes those properties.
The case-study table describes the latest completed v2 run, rather than an
aggregate over repeats. Scores verify against the frozen labels; this is not a
fresh semantic proof of every benchmark answer.

Cost fields use the captured pricing assumptions and conservative reservations
for unknown usage. They are accounting records or estimates, not invoices.
Upstream material retains the terms and attribution in `LICENSES/`.
