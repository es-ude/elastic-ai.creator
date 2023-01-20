from abc import ABC, abstractmethod
from typing import Collection, Iterable

from elasticai.creator.vhdl.code import CodeFile, CodeModule
from elasticai.creator.vhdl.port import Port
from elasticai.creator.vhdl.vhdl_files import VHDLFile


class BaseVHDLDesign(CodeModule, ABC):
    @abstractmethod
    def get_port(self) -> Port:
        ...

    @property
    def name(self) -> str:
        return self._name

    @property
    def files(self) -> Collection[CodeFile]:
        return self._templates.values()

    @property
    def submodules(self) -> Collection["CodeModule"]:
        return []

    def __init__(
        self,
        name: str,
        template_names: Iterable[str],
    ):
        self._name = name
        self._templates = {name: VHDLFile(name) for name in template_names}
