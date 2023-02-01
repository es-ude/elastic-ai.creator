import dataclasses
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol


class Signal(Protocol):
    @abstractmethod
    def accepts(self, other: "Signal") -> bool:
        ...

    @abstractmethod
    def id(self) -> str:
        ...

    @abstractmethod
    def definition(self, prefix: str = "") -> str:
        ...


@dataclass(kw_only=True)
class _SignalConfiguration:
    id: str
    default: str
    accepted_names: list[str]


@dataclass(kw_only=True)
class _VectorSignalConfiguration(_SignalConfiguration):
    width: int


class SignalBuilder:
    def __init__(self):
        self.args = {
            "width": 0,
            "id": "x",
            "accepted_names": list(),
            "default": None,
        }

    def width(self, width: int) -> "SignalBuilder":
        self.args.update({"width": width})
        return self

    def id(self, name: str) -> "SignalBuilder":
        self.args.update({"id": name})
        return self

    def accepted_names(self, names: list[str]) -> "SignalBuilder":
        self.args.update({"accepted_names": names})
        return self

    def default(self, value: str) -> "SignalBuilder":
        self.args.update({"default": value})
        return self

    def build(self) -> Signal:
        if self.args["width"] > 0:
            return _VectorSignalImpl(_VectorSignalConfiguration(**self.args))
        else:
            args_without_width = filter(
                lambda entry: entry[0] != "width", self.args.items()
            )
            return _SignalImpl(_SignalConfiguration(**dict(args_without_width)))


class _SignalImpl(Signal):
    def __init__(self, base: _SignalConfiguration):
        self._base = base

    def id(self) -> str:
        return self._base.id

    def _default(self) -> str:
        if self._base.default is None:
            return ""
        return f" := {self._base.default}"

    def definition(self, prefix: str = "") -> str:
        return f"signal {prefix}{self.id()} : std_logic{self._default()};"

    def accepts(self, other: Signal) -> bool:
        if isinstance(other, _SignalImpl):
            return other._base.id in self._base.accepted_names
        return False


class _VectorSignalImpl(_SignalImpl):
    def __init__(self, base: _VectorSignalConfiguration) -> None:
        super().__init__(base)

    def definition(self, prefix: str = "") -> str:
        return (
            f"signal {prefix}{self.id()} :"
            f" std_logic_vector({self._base.width - 1} downto 0){self._default()};"
        )

    def accepts(self, other: "_VectorSignalImpl") -> bool:
        self_base = typing.cast(_VectorSignalConfiguration, self._base)
        other_base = typing.cast(_VectorSignalConfiguration, other._base)
        return super().accepts(other) and self_base.width == other_base.width
