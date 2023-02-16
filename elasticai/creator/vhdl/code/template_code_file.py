from abc import abstractmethod
from typing import Iterable, Protocol, Union

from elasticai.creator.vhdl.code.code_file import CodeFile


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
