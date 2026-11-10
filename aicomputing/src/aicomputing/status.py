"""Shared result statuses with stable string values for JSON, CSV, and callers."""

from enum import Enum


class _StringStatus(str, Enum):
    def __str__(self) -> str:
        return self.value


class ParseStatus(_StringStatus):
    """How a prediction reply yielded a value; None means unrecorded/inapplicable."""

    PARSED = "parsed"
    FALLBACK = "fallback_parsed"
    FAILED = "failed"


class ValidationStatus(_StringStatus):
    """Final output-checking outcome; these labels do not establish correctness."""

    VALID = "valid"
    NOT_CHECKED = "not_checked"
    INVALID_TYPE = "invalid_type"
    INVALID_CONSTRAINT = "invalid_constraint"
    UNPARSED = "unparsed"
    EXECUTION_FAILED = "execution_failed"
    UNAVAILABLE = "unavailable"
