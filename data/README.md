# Reviewed benchmark dataset

[`dataset-release.json`](dataset-release.json) is the shared, versioned input
to the evaluation command. It contains reviewed specifications, frozen typed
inputs and expected answers, scoring profiles, review notes and quarantine
decisions. A normal evaluation uses this file directly, without downloading
datasets or executing reference implementations.

Release: `specification-output-reviewed-v3`.
Size: 3,939,750 bytes. SHA256:

```text
aabacacd3d16c45e74bccadbde1e412fae36b2218d47d607049ddb719ef3e717
```

| Benchmark | Source tasks | Approved | Quarantined |
| --- | ---: | ---: | ---: |
| HumanEval+ | 164 | 153 | 11 |
| MBPP+ | 378 | 343 | 35 |
| GSM8K | 1,319 | 1,181 | 138 |

Selection uses approved tasks only. The release fixes the reviewed input for
each task; the command's seed and sample size select tasks from those pools.
Dataset and selection fingerprints are recorded separately in each run. See
the [benchmark reference](../docs/evaluation.md) for grading and MBPP published
target details.

MBPP+ has 342 observations matched to published literal targets. The remaining
approved observation, `Mbpp/630`, is explicitly marked as a specification-reviewed
target because its published test script has no precomputed output list.

Per-task review rationales are embedded in the release. Provenance uses portable
source identifiers, checksums and published-target IDs. The release contains
only data and curation decisions; selection settings belong to the evaluator.
Git attributes preserve the release bytes across operating systems.
For a different reviewed release, pass `--dataset-release path/to/release.json`.

## Sources and notices

| Included material | Upstream attribution | Upstream terms |
| --- | --- | --- |
| HumanEval+ | [EvalPlus HumanEval+](https://huggingface.co/datasets/evalplus/humanevalplus), based on [OpenAI HumanEval](https://github.com/openai/human-eval) | EvalPlus Apache-2.0; original HumanEval MIT |
| MBPP+ | [EvalPlus MBPP+](https://huggingface.co/datasets/evalplus/mbppplus), based on [Google Research MBPP](https://github.com/google-research/google-research/tree/master/mbpp), Austin et al., 2021 | EvalPlus Apache-2.0; original MBPP CC BY 4.0 |
| GSM8K | [OpenAI GSM8K](https://github.com/openai/grade-school-math), Cobbe et al., 2021 | MIT |

Copies of the applicable Apache and MIT notices, source links and the CC BY 4.0
license link are retained in [`THIRD_PARTY_NOTICES.txt`](THIRD_PARTY_NOTICES.txt).
The original MBPP data terms are stated in the
[Google Research repository README](https://github.com/google-research/google-research/blob/master/README.md).

This project adapted the upstream material into a typed JSON release, wrapped
public specifications for output prediction, reviewed and froze input/target
pairs, assigned scoring profiles and quarantined unresolved cases. These
project-specific changes and original artifact hashes are recorded in the
release. The benchmark data retains its upstream terms separately from the
package code's MIT license.
