"""Load the archived, author-released TDMSci CoNLL splits without changing labels."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .contract import EntityRecord, FIELDS, LABELS, Sentence

DATA_DIR = Path(__file__).with_name("data")
FILES = {"train": ("train_1500_v2.conll", "8107bc8a0214f3c57d234e2ab69958c1d4d8b11e43e2007d97ea42efaf7a231a"),
         "test": ("test_500_v2.conll", "5d776e10f080c2872bba416716bc2109a86946974942a255b8dda3dd78078559")}


@dataclass(frozen=True)
class Case:
    id: str
    split: str
    sentence: Sentence
    gold: EntityRecord


def load_split(split: str) -> list[Case]:
    if split not in FILES:
        raise ValueError("Choose the train or test split")
    filename, expected_hash = FILES[split]
    payload = (DATA_DIR / filename).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f"TDMSci {split} file does not match the archived release")
    cases = []
    entries: list[tuple[str, int, str]] = []

    def finish():
        if not entries:
            return
        sentence = Sentence.from_conll(entries)
        groups = {name: [] for name in FIELDS}
        current_field = None
        start = 0
        for index, (_, _, tag) in enumerate(entries + [("", 0, "O")]):
            if tag.startswith("I-"):
                if current_field is None or LABELS[current_field] != tag[2:]:
                    raise ValueError(f"Invalid I-tag in TDMSci {split} sentence {len(cases) + 1}")
                continue
            if current_field is not None:
                groups[current_field].append({"start": start, "end": index})
            current_field = next((name for name, label in LABELS.items() if tag == "B-" + label), None)
            if tag != "O" and current_field is None:
                raise ValueError(f"Unknown TDMSci label {tag}")
            start = index
        gold = EntityRecord.from_dict(groups, sentence)
        cases.append(Case(f"{split[0].upper()}{len(cases) + 1:04d}", split, sentence, gold))
        entries.clear()

    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            finish()
            continue
        parts = line.split("\t")
        if len(parts) != 3 or not parts[1].isdigit():
            raise ValueError(f"Malformed TDMSci {split} row")
        entries.append((parts[0], int(parts[1]), parts[2]))
    finish()
    return cases
