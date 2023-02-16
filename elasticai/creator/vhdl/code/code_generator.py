from abc import abstractmethod
from typing import Callable, Protocol, runtime_checkable

from .code import Code


@runtime_checkable
class CodeGenerator(Protocol):
    @abstractmethod
    def code(self) -> Code:
        ...


CodeGeneratorCompatible = Code | CodeGenerator | str | Callable[[], Code]
