"""The versioned, provider-independent prediction response protocol."""

ANSWER_FORMAT = "framed_answer_v1"
FINAL_START = "BEGIN FINAL ANSWER"
FINAL_END = "END FINAL ANSWER"
def frame_answer(payload: str) -> str:
    return f"{FINAL_START}\n{payload}\n{FINAL_END}"
