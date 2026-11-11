"""Small text-only interface for tagging research sentences."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json

import streamlit as st

from . import conventional, model
from .contract import FIELDS, LABELS, MAX_TEXT_CHARS, SPECIFICATION, Sentence

COLORS = {"TASK": "#d7eaff", "DATASET": "#dff4df", "METRIC": "#ffe7c7"}


def mention_rows(results):
    rows = []
    for engine in ("rules", "model"):
        entry = results.get(engine)
        if entry is None or entry["outcome"]["status"] != "completed":
            continue
        for field in FIELDS:
            for mention in entry["outcome"]["record"][field]:
                rows.append({"Method": "Rules" if engine == "rules" else "AI",
                             "Type": LABELS[field].title(), "Text": mention["text"],
                             "Start token": mention["start"], "End token": mention["end"]})
    return rows


def highlighted_sentence(sentence: Sentence, record):
    spans = []
    for field in FIELDS:
        for mention in record[field]:
            start = sentence.offsets[mention["start"]][0]
            end = sentence.offsets[mention["end"] - 1][1]
            spans.append((start, end, LABELS[field]))
    parts = []
    cursor = 0
    for start, end, label in sorted(spans):
        parts.append(html.escape(sentence.text[cursor:start]))
        parts.append(f'<mark style="background:{COLORS[label]};padding:2px 4px">'
                     f'{html.escape(sentence.text[start:end])}</mark>'
                     f'<sup style="margin-right:5px">{label.title()}</sup>')
        cursor = end
    parts.append(html.escape(sentence.text[cursor:]))
    return '<div style="line-height:2.1;white-space:pre-wrap">' + "".join(parts) + "</div>"


def csv_text(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["Method", "Type", "Text", "Start token", "End token"])
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main():
    st.set_page_config(page_title="Research sentence tagger", layout="wide")
    st.title("Research sentence tagger")
    st.write("Paste one research sentence. The app marks mentions of research tasks, datasets, and evaluation metrics.")
    try:
        settings = model.Settings.from_env()
    except ValueError:
        settings = None
        st.warning("Model configuration is invalid; the offline rules remain available.")
    model_ready = settings is not None and settings.has_credentials()
    text = st.text_area("Sentence", height=140, max_chars=MAX_TEXT_CHARS,
                        placeholder="We evaluate named entity recognition on CoNLL-2003 using F1 score.")
    sentence = None
    if text.strip():
        try:
            sentence = Sentence.from_text(text)
        except ValueError as exc:
            st.warning(str(exc))
    if sentence is not None:
        st.caption(f"{len(sentence.tokens)} tokens. Both methods receive the same token sequence.")
    st.caption(f"AI uses {settings.model} and incurs API charges." if model_ready
               else "AI access is not configured. Rules work offline.")
    with st.form("extract"):
        rules_clicked = st.form_submit_button("Run rules (offline)", disabled=sentence is None, type="primary")
        model_clicked = st.form_submit_button("Run AI", disabled=sentence is None or not model_ready)
    if rules_clicked or model_clicked:
        fingerprint = hashlib.sha256(sentence.text.encode("utf-8")).hexdigest()
        if st.session_state.get("tagged_text_sha256") != fingerprint:
            st.session_state["tagger_results"] = {}
            st.session_state["tagged_text_sha256"] = fingerprint
        try:
            if rules_clicked:
                outcome = conventional.extract(sentence)
            else:
                with st.spinner("Tagging research entities..."):
                    with settings.runtime() as runtime:
                        outcome = model.extract(sentence, runtime, settings)
        except Exception as exc:
            st.error(f"Extraction failed during setup ({type(exc).__name__}).")
            return
        st.session_state["tagger_results"][outcome.engine] = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "text": sentence.text, "tokens": list(sentence.tokens),
            "outcome": outcome.to_dict(),
            "settings": settings.public() if model_clicked else {"algorithm": "training-phrase-lexicon-v1"},
            "estimated_api_cost_usd": model.estimated_cost(outcome, settings) if model_clicked else 0.0}

    results = st.session_state.get("tagger_results", {})
    if results:
        st.divider()
        st.subheader("Tagged sentence")
        st.caption("Results refer to the last submitted text. Token ranges use zero-based, end-exclusive indices.")
        for engine in ("rules", "model"):
            entry = results.get(engine)
            if entry is None:
                continue
            label = "Rules" if engine == "rules" else "AI"
            outcome = entry["outcome"]
            st.markdown(f"**{label}** · {outcome['status'].replace('_', ' ')} · {outcome['elapsed_seconds']:.3f}s")
            if outcome["status"] == "completed":
                displayed = Sentence.from_text(entry["text"])
                st.markdown(highlighted_sentence(displayed, outcome["record"]), unsafe_allow_html=True)
            else:
                st.error(f"{label} did not produce a valid result.")
            for warning in outcome["warnings"]:
                st.warning(f"{label}: {warning}")
            if engine == "model" and entry["estimated_api_cost_usd"] is not None:
                st.caption(f"Estimated AI cost: USD {entry['estimated_api_cost_usd']:.6f}")
        rows = mention_rows(results)
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.info("No entities were found.")
        st.download_button("Download findings (CSV)", csv_text(rows), "research-entities.csv", "text/csv")
        st.download_button("Download complete result (JSON)",
                           json.dumps(results, ensure_ascii=True, indent=2),
                           "research-entities.json", "application/json")
        if "model" in results and results["model"]["outcome"]["prediction"] is not None:
            with st.expander("AI request and raw response"):
                st.json(results["model"]["outcome"]["prediction"])
    with st.expander("Extraction specification"):
        st.text(SPECIFICATION)
