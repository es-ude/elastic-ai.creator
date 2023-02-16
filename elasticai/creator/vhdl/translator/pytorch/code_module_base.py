import dataclasses
import typing

from elasticai.creator.vhdl.code.code_file import CodeFile
from elasticai.creator.vhdl.code.code_module import CodeModule

T_CodeFile = typing.TypeVar("T_CodeFile", bound=CodeFile)
T_CodeModuleBase = typing.TypeVar("T_CodeModuleBase", bound="CodeModuleBase")


@dataclasses.dataclass
class CodeModuleBase(CodeModule[T_CodeFile]):
    @property
    def submodules(self: T_CodeModuleBase) -> typing.Collection[T_CodeModuleBase]:
        return self._submodules

    @property
    def files(self) -> typing.Collection[T_CodeFile]:
        return self._files

    @property
    def name(self) -> str:
        return self._name

    def __init__(
        self,
        name: str,
        files: typing.Collection[T_CodeFile],
        submodules: typing.Collection[T_CodeModuleBase] = tuple(),
    ):
        self._name = name
        self._files = files
        self._submodules = submodules
