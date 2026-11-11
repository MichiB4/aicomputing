"""A deterministic phrase matcher built only from TDMSci training labels."""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import time

from .contract import EntityRecord, Extraction, FIELDS, LABELS, Sentence
from .dataset import load_split


@lru_cache(maxsize=1)
def training_lexicon():
    """Choose each phrase's most common training label; never read test labels."""
    counts = defaultdict(Counter)
    for case in load_split("train"):
        for label, start, end in case.gold.spans():
            phrase = tuple(token.casefold() for token in case.sentence.tokens[start:end])
            counts[phrase][label] += 1
    return {phrase: sorted(scores, key=lambda label: (-scores[label], label))[0]
            for phrase, scores in counts.items()}


def extract(sentence: Sentence, *, lexicon=None) -> Extraction:
    started = time.perf_counter()
    glossary = training_lexicon() if lexicon is None else lexicon
    max_length = max((len(phrase) for phrase in glossary), default=0)
    tokens = tuple(token.casefold() for token in sentence.tokens)
    groups = {name: [] for name in FIELDS}
    field_for_label = {label: name for name, label in LABELS.items()}
    index = 0
    while index < len(tokens):
        match = next(((length, glossary[tokens[index:index + length]])
                      for length in range(min(max_length, len(tokens) - index), 0, -1)
                      if tokens[index:index + length] in glossary), None)
        if match is None:
            index += 1
            continue
        length, label = match
        groups[field_for_label[label]].append({"start": index, "end": index + length})
        index += length
    record = EntityRecord.from_dict(groups, sentence)
    return Extraction("rules", "completed", record, time.perf_counter() - started)
