from abc import ABC, abstractmethod
from typing import Collection, Iterable, Mapping, Protocol

from elasticai.creator.vhdl.code import CodeFile, CodeModule
from elasticai.creator.vhdl.designs.vhdl_files import VHDLFile
from elasticai.creator.vhdl.ports import Port, PortMap


class VHDLDesign(CodeModule, Protocol):
    @abstractmethod
    def get_port(self) -> Port:
        ...

    @abstractmethod
    def get_port_map(self, prefix: str) -> PortMap:
        ...

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


class BaseVHDLDesign(VHDLDesign, CodeModule, ABC):
    @abstractmethod
    def get_port(self) -> Port:
        ...

    def get_port_map(self, prefix: str) -> PortMap:
        return self.get_port().build_port_map(prefix)

    @property
    def name(self) -> str:
        return self._name

    @property
    def files(self) -> Iterable[VHDLFile]:
        return self._templates

    @property
    def submodules(self) -> Collection["CodeModule"]:
        return []

    def __init__(
        self,
        name: str,
        files: Iterable[VHDLFile],
    ):
        self._name = name
        self._templates = files
