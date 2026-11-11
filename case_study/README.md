# Research sentence tagger

This separate case study uses `aicomputing` to identify mentions of **research
tasks**, **datasets**, and **evaluation metrics** in a pasted sentence. The
Streamlit app highlights the returned text spans. An offline phrase-matching
baseline provides a conventional comparison. Both paths receive the same
tokens and return the same span schema.

The evaluation uses the independently annotated [TDMSci corpus](https://aclanthology.org/2021.eacl-main.59/).
Its [archived author-released files](src/paper_parser/data/README.md), license,
and checksums are bundled with this package. The files contain short sentences
from NLP papers, so the study requires neither PDF processing nor our own gold
annotations. They label entity mentions, not numerical results or relationships
among them.

## Setup and app

From the repository root:

```powershell
python -m venv case_study/.venv
case_study/.venv/Scripts/python -m pip install -e "./aicomputing[openai]" -e "./case_study"
cd case_study
.venv/Scripts/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
```

Open `http://127.0.0.1:8502`. Paste a sentence such as `We evaluate named
entity recognition on CoNLL-2003 using F1 score.` The app accepts one text
input of up to 2,500 characters. Run the offline rules or AI method, inspect
the colored spans, and download the current findings as CSV or JSON.
On Linux/macOS, use `.venv/bin/python`.

Rules mode works without credentials. For AI, set `OPENAI_API_KEY` or
`PAPER_PARSER_KEY_FILE` to a local file with one `OPENAI_API_KEY=...`
assignment. `.env.example` documents optional settings; it is not loaded
automatically. The default is GPT-6.1 Terra, direct strategy, native reasoning
`none`, strict validation, no cache or automatic retries, and a 120-second
HTTP phase timeout. The default request omits a provider output-token cap
because Terra rejects that parameter. Other models must support the requested
settings. The app makes an API call only when **Run AI** is selected.

## Evaluation

This command evaluates the baseline on the complete published test split with
no API calls:

```powershell
.venv/Scripts/python -m paper_parser.study --engine rules --output output/tdmsci-rules
```

This paired command uses the same 100 sampled sentences for both methods and
makes paid model calls:

```powershell
.venv/Scripts/python -m paper_parser.study --engine both --sample-size 100 --seed 42 --key-file "C:\path\to\credentials.env" --output output/tdmsci-paired-100
```

Use `--case-id T0001` to select a test sentence, `--split train` for training
data, and `--model`, `--timeout`, `--input-price`, and `--output-price` to
configure a run. The baseline builds its phrase lexicon from **training labels
only**; model prompts contain tokens and the natural-language specification,
never gold labels. Both methods are graded against exact token spans and entity
types. The study reports micro precision, recall, F1, per-type scores, exact
sentence accuracy, failures, latency, usage, and estimated cost when prices
are supplied. Invalid outputs count as missed gold mentions.

Each study writes one detailed CSV and one JSON under a new basename. Both
contain gold and predicted spans, prompts, raw model responses, settings, and
source hashes. Existing files are not overwritten. Infrastructure failures
stop the batch after checkpointing the affected case; there are no automatic
retries or resume operation.

The released v2 files parse into 1,523 training and 487 test sentence blocks,
though their names and the original article report 1,500/500. Their
entity counts match the article. The app tokenizes arbitrary pasted text with
a regular expression; study cases use the corpus's own tokens.

### Recorded 100-sentence run and duplicate sensitivity

The existing paired run used seed 42 and retained 100 sampled test
rows exactly as released. Its detailed CSV and JSON are stored locally under
`output/` and are excluded from the GitHub repository. The original 100-row
scores remain the primary results. The released corpus has repeated token
sequences within and across splits; counts and annotation conflicts are
documented in the
[dataset provenance](src/paper_parser/data/README.md).

In the 100-row sample, `T0016` and `T0064` have identical tokens and labels.
`T0173`, `T0175`, and `T0177` exactly match training sentences, with the same
gold annotations. For a **sensitivity analysis only**, exclude those three
training-overlap rows and retain the first occurrence (`T0016`) of the
within-test pair. This leaves 96
test sentences. The 96-row scores below were recomputed from the saved 100-row
predictions; they required no new model calls and do not replace the original
run. Duplicate identity means exact, case-sensitive ordered token equality.

| Method and set | Precision | Recall | Micro F1 | Exact sentences |
| --- | ---: | ---: | ---: | ---: |
| Terra, original 100 | 54.5% | 69.5% | 61.1% | 48/100 (48.0%) |
| Terra, duplicate sensitivity 96 | 54.8% | 69.9% | 61.4% | 46/96 (47.9%) |
| Phrase matcher, original 100 | 46.2% | 43.9% | 45.0% | 29/100 (29.0%) |
| Phrase matcher, duplicate sensitivity 96 | 44.9% | 42.3% | 43.6% | 26/96 (27.1%) |

The small change in Terra's F1 suggests that these four rows do not drive the
reported result. It does not test whether the public dataset appeared in the
model's pretraining material. Report the 100-row result first and the 96-row
calculation as a sensitivity check.

### Specification and annotation guidelines

The model run used the exact `tdmsci-token-spans-v1` specification saved in its
result JSON and in [`contract.py`](src/paper_parser/contract.py). A review of
the authors' [annotation guidelines, Appendix A](https://aclanthology.org/2021.eacl-main.59.pdf#page=7)
found the following alignment and gaps:

| Guideline | Evaluated specification |
| --- | --- |
| Identify Task problems, Dataset corpora/resources, and Metric measures in sentences from any paper section. | Broadly aligned. The prompt includes mentions of prior work, but describes datasets as corpora, benchmarks, or test collections without naming language resources. |
| Annotate factual, content-bearing, self-sufficient mentions; omit anonymous references such as “this task” or “the dataset.” | Not stated. The prompt's “every relevant mention” may encourage extra generic spans. |
| Choose the minimum meaningful span and omit determiners. Include `corpus`/`dataset`/`benchmark` in a dataset span when it is the head noun; usually omit `task`/`problem` from a task span. | Only the general instruction to omit surrounding articles and prepositions is present. For example, in saved case `T0024` the model marked `neural machine translation`, while the gold span is `machine translation`. The prompt leaves that boundary choice open. |
| Interpret shared-task names by their actual referent in context and include higher-level tasks. Keep full names with adjacent abbreviations; handle conjunctions and task premodifiers according to the annotation examples. | These decisions are not specified explicitly. |
| The released CoNLL data keeps the longest entity where original annotations overlap. | The prompt forbids overlapping output, matching the flat CoNLL target, but does not tell the model which span to retain. |

For a future prompt version, explicitly describe those rules and test it in a
new run. Changing the prompt now would make the saved 100-sentence results no
longer representative of the current specification. The present F1 measures
agreement with the released labels under the evaluated prompt; it is not a
measure of general extraction quality outside this corpus.
