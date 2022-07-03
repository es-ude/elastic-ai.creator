from abc import ABC, abstractmethod
from typing import Optional

from elasticai.creator.resource_utils import (
    Package,
    PathType,
    read_text,
    read_text_from_path,
)
from elasticai.creator.vhdl.language import Code


class VHDLComponent(ABC):
    @property
    @abstractmethod
    def file_name(self) -> str:
        ...

    @abstractmethod
    def __call__(self, custom_template: Optional[PathType] = None) -> Code:
        ...


class VHDLStaticComponent(VHDLComponent):
    def __init__(self, template_package: Package, file_name: str) -> None:
        self._template_package = template_package
        self._file_name = file_name

    @property
    def file_name(self) -> str:
        return self._file_name

    def __call__(self, custom_template: Optional[PathType] = None) -> Code:
        if custom_template is None:
            code = read_text(self._template_package, self._file_name)
        else:
            code = read_text_from_path(custom_template)

        yield from code.splitlines()
