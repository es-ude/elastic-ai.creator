import dataclasses
from abc import abstractmethod
from collections.abc import Collection
from typing import Callable, Iterable, Optional, Protocol, Union, runtime_checkable

Code = Iterable[str]


@runtime_checkable
class CodeGenerator(Protocol):
    @abstractmethod
    def code(self) -> Code:
        ...


CodeGeneratorCompatible = Code | CodeGenerator | str | Callable[[], Code]


class Translatable(Protocol):
    @abstractmethod
    def translate(self) -> "CodeModule":
        ...


class CodeFile(Protocol):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def code(self) -> Code:
        ...


class TemplateCodeFile(CodeFile, Protocol):
    @property
    @abstractmethod
    def multi_line_parameters(self) -> dict[str, Iterable[str]]:
        ...

    @property
    @abstractmethod
    def single_line_parameters(self) -> dict[str, str]:
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Union[str, Iterable[str]]]:
        ...


class CodeModule(Protocol):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def files(self) -> Collection[CodeFile]:
        ...

    @property
    @abstractmethod
    def submodules(self) -> Collection["CodeModule"]:
        ...


@dataclasses.dataclass
class CodeModuleBase(CodeModule):
    @property
    def submodules(self) -> Collection["CodeModule"]:
        return self._submodules

    @property
    def files(self) -> Collection[CodeFile]:
        return self._files

    @property
    def name(self) -> str:
        return self._name

    def __init__(
        self,
        name: str,
        files: Collection[CodeFile],
        submodules: Collection["CodeModule"] = tuple(),
    ):
        self._name = name
        self._files = files
        self._submodules = submodules


class CodeFileBase(CodeFile):
    @property
    def name(self) -> str:
        return self._name

    def code(self) -> Code:
        return self._code

    def __repr__(self) -> str:
        return f"CodeBaseFile(name={self._name}, code={self._code})"

    def __init__(self, name: str, code: Code):
        self._name = name
        self._code = code
