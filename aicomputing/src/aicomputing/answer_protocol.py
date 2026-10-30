"""Versioned, provider-independent prediction response formats."""

LEGACY = "legacy"
FRAMED_V1 = "framed_v1"
FRAMED_TEXT_V1 = "framed_text_v1"
FINAL_START = "BEGIN FINAL ANSWER"
FINAL_END = "END FINAL ANSWER"
ANSWER_FORMATS = (LEGACY, FRAMED_V1, FRAMED_TEXT_V1)


def validate_answer_format(value: str) -> str:
    if not isinstance(value, str) or value not in ANSWER_FORMATS:
        raise ValueError(f"answer_format must be one of {ANSWER_FORMATS}")
    return value


def frame_answer(payload: str) -> str:
    return f"{FINAL_START}\n{payload}\n{FINAL_END}"
