from abc import abstractmethod
from typing import Protocol


class Identifiable(Protocol):
    @abstractmethod
    def id(self) -> str:
        ...
