from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from elasticai.creator.resource_utils import Package, read_text
from elasticai.creator.vhdl.language import Code


class VHDLFile(Protocol):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def code(self) -> Code:
        ...


class StaticVHDLFile(VHDLFile):
    def __init__(self, template_package: Package, file_name: str) -> None:
        self._template_package = template_package
        self._file_name = file_name

    @property
    def name(self) -> str:
        return self._file_name

    @property
    def code(self) -> Code:
        code = read_text(self._template_package, self._file_name)
        yield from code


class VHDLModule(Protocol):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def files(self) -> Iterable[VHDLFile]:
        ...


class VHDLBaseModule(VHDLModule):
    @property
    def files(self) -> Iterable[VHDLFile]:
        yield from self._files

    @property
    def name(self) -> str:
        return self._name

    def __init__(self, name: str, files: Iterable[VHDLFile]):
        self._name = name
        self._files = files


@dataclass
class VHDLBaseFile(VHDLFile):
    @property
    def name(self) -> str:
        return self._name

    @property
    def code(self) -> Code:
        return self._code

    def __repr__(self) -> str:
        return f"VHDLBaseFile(name={self._name}, code={self._code})"

    def __init__(self, name: str, code: Code):
        self._name = name
        self._code = code
