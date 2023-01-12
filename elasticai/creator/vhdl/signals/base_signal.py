from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, TypeVar

from elasticai.creator.vhdl.code import Code
from elasticai.creator.vhdl.connectable import Connectable


class Identifiable(Protocol):
    @abstractmethod
    def id(self) -> str:
        ...


class Signal(Identifiable, Protocol):
    @abstractmethod
    def definition(self) -> str:
        ...


class OutSignal(Signal, Protocol):
    ...


class InSignal(Signal, Connectable, Protocol):
    @abstractmethod
    def code(self) -> Code:
        ...


T_Reversible = TypeVar("T_Reversible", bound="Reversible", covariant=True)


class Reversible(Protocol[T_Reversible]):
    def reverse(self) -> T_Reversible:
        ...


class BaseInSignal(InSignal, ABC):
    def __init__(self) -> None:
        self._connected_signal: Optional[Identifiable] = None

    def is_missing_inputs(self) -> bool:
        return self._connected_signal is None

    @abstractmethod
    def matches(self, other: Any) -> bool:
        ...

    def connect(self, other: Identifiable) -> None:
        if self.matches(other):
            self._connected_signal = other

    def code(self) -> Code:
        if self._connected_signal is not None:
            return [f"{self.id()} <= {self._connected_signal.id()};"]
        else:
            return []
