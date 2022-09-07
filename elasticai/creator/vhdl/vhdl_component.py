from typing import Any, Iterable, Protocol

from elasticai.creator.resource_utils import Package, read_text
from elasticai.creator.vhdl.language import Code


class VHDLComponent(Protocol):
    @property
    def file_name(self) -> str:
        return ""

    def __call__(self) -> Code:
        ...


class VHDLStaticComponent:
    def __init__(self, template_package: Package, file_name: str) -> None:
        self._template_package = template_package
        self._file_name = file_name

    @property
    def file_name(self) -> str:
        return self._file_name

    def __call__(self) -> Code:
        code = read_text(self._template_package, self._file_name)
        yield from code.splitlines()


class VHDLModule(Protocol):
    def components(self, args: Any) -> Iterable[VHDLComponent]:
        ...
