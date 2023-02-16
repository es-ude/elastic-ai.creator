from abc import abstractmethod
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
