"""Native reasoning controls advertised or declared for one model endpoint."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningCapabilities:
    backend: str
    model: str
    parameter: str | None
    endpoint: str | None = None
    support: str = "unknown"
    values: tuple[bool | str, ...] = ()
    default: bool | str | None = None
    source: str = "user_declared"

    def __post_init__(self) -> None:
        for name in ("backend", "model", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.parameter is not None and (
            not isinstance(self.parameter, str) or not self.parameter.strip()
        ):
            raise ValueError("parameter must be a nonempty string or None")
        if self.endpoint is not None and (
            not isinstance(self.endpoint, str) or not self.endpoint.strip()
        ):
            raise ValueError("endpoint must be a nonempty string or None")
        if self.support not in ("unknown", "supported", "unsupported"):
            raise ValueError("Invalid capability support state")
        if type(self.values) not in (tuple, list):
            raise TypeError("values must be a tuple or list")
        object.__setattr__(self, "values", tuple(self.values))
        for value in self.values:
            self._check_value(value)
        if len({(type(value), value) for value in self.values}) != len(self.values):
            raise ValueError("Capability values must be unique")
        if self.default is not None:
            self._check_value(self.default)
        if self.support == "supported":
            if self.parameter is None or not self.values:
                raise ValueError("Supported controls require a parameter and values")
            if self.default is not None and not self.accepts(self.default):
                raise ValueError("Advertised default is not a supported value")
        elif self.values or self.default is not None:
            raise ValueError("Unknown/unavailable controls cannot declare values")

    @staticmethod
    def _check_value(value: object) -> None:
        if type(value) not in (bool, str):
            raise TypeError("Native controls must be booleans or strings")
        if isinstance(value, str) and not value.strip():
            raise ValueError("Native control strings must not be empty")

    def accepts(self, value: object) -> bool:
        return any(type(value) is type(item) and value == item for item in self.values)

    @property
    def kind(self) -> str:
        """Describe the offered controls, not the model's internal behavior."""
        if self.support != "supported":
            return self.support
        if len(self.values) == 1:
            return "fixed"
        if len(self.values) == 2 and self.accepts(False) and self.accepts(True):
            return "toggle"
        return "levels" if all(type(value) is str for value in self.values) else "choices"

    def validate(self, value: bool | str | None, *, allow_unknown: bool = False) -> None:
        if type(allow_unknown) is not bool:
            raise TypeError("allow_unknown must be a bool")
        if value is None:
            return
        self._check_value(value)
        if self.support == "unknown":
            if not allow_unknown:
                raise ValueError("Reasoning support is unknown; set allow_unknown=True for an experimental request")
        elif self.support == "unsupported" or not self.accepts(value):
            raise ValueError(f"Unsupported native reasoning value: {value!r}")
