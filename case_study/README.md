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
HTTP phase timeout. The default request omits an application output-token cap.
Other models must support the requested
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

JSON preserves model text exactly. CSV escapes invalid Unicode characters so
an invalid model response cannot prevent checkpointing the study.

The released v2 files parse into 1,523 training and 487 test sentence blocks,
though their names and the original article report 1,500/500. Their
entity counts match the article. The app tokenizes arbitrary pasted text with
a regular expression; study cases use the corpus's own tokens.

## Extraction specification and dataset limits

The app and study use the versioned specification in
[`contract.py`](src/paper_parser/contract.py), based on the authors'
[annotation guidelines, Appendix A](https://aclanthology.org/2021.eacl-main.59.pdf#page=7).
It defines entity categories, span boundaries, abbreviations, conjunctions,
and the flat output schema. Each result records the exact specification and
its revision.

The corpus contains repeated sentences and train/test overlap, including
conflicting annotations. Counts are documented in the
[dataset provenance](src/paper_parser/data/README.md). Sampling retains the
released rows and labels. Scores therefore measure agreement with this corpus;
they do not establish performance on unseen papers or general extraction quality.
