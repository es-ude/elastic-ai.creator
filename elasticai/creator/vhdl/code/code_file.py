from abc import abstractmethod
from typing import Protocol

from .code import Code


class CodeFile(Protocol):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def lines(self) -> Code:
        ...
