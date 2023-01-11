from abc import ABC, abstractmethod
from typing import (
    Any,
    ClassVar,
    Collection,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from elasticai.creator.vhdl.connectable import Connectable


class SignalProvider(Protocol):
    def in_signals(self) -> Collection["InSignal"]:
        ...

    def out_signals(self) -> Collection["OutSignal"]:
        ...


T_Signal_contra = TypeVar("T_Signal_contra", bound="Signal", contravariant=True)


class Identifiable(Protocol):
    @abstractmethod
    def id(self) -> str:
        ...


class Signal(Protocol):
    @abstractmethod
    def definition(self) -> str:
        ...


@runtime_checkable
class InSignal(Signal, Connectable, Protocol):
    @abstractmethod
    def code(self) -> list[str]:
        ...


class NullIdentifiable(Identifiable):
    def id(self) -> str:
        return ""


class BaseInSignal(InSignal, ABC):
    _NULL_SIGNAL: ClassVar[Identifiable] = NullIdentifiable()

    def __init__(self, id: str):
        self._out_signal: Identifiable = self._NULL_SIGNAL
        self._id = id

    def code(self) -> list[str]:
        return [f"{self.id()} <= {self._out_signal.id()}"]

    @abstractmethod
    def definition(self) -> str:
        return ""

    def id(self) -> str:
        return self._id

    def is_missing_inputs(self) -> bool:
        return self._out_signal == self._NULL_SIGNAL

    @abstractmethod
    def matches(self, other: Any) -> bool:
        ...

    def connect(self, other: Identifiable) -> None:
        if self.matches(other):
            self._out_signal = other


@runtime_checkable
class OutSignal(Signal, Protocol):
    ...
