from abc import abstractmethod
from collections.abc import Collection
from typing import Iterable, Callable, Protocol

Code = Iterable[str]
CodeGenerator = Callable[[], Code]
CodeGeneratorCompatible = Code | CodeGenerator | str


class CodeModule(Protocol):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def submodules(self) -> Collection["CodeModule"]:
        ...

    @property
    @abstractmethod
    def files(self) -> Collection["CodeFile"]:
        ...

    @abstractmethod
    def save_to(self, directory: str) -> None:
        ...


class CodeFile(Protocol):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def save_to(self, directory: str) -> None:
        ...

    @abstractmethod
    def code(self) -> Code:
        ...


class Translatable(Protocol):
    @abstractmethod
    def translate(self) -> CodeModule:
        ...
