import dataclasses
import typing

T_CodeModuleBase = typing.TypeVar("T_CodeModuleBase", bound="CodeModuleBase")


@dataclasses.dataclass
class CodeModuleBase:
    @property
    def submodules(self: T_CodeModuleBase) -> typing.Collection[T_CodeModuleBase]:
        return self._submodules

    @property
    def files(self) -> typing.Collection:
        return self._files

    @property
    def name(self) -> str:
        return self._name

    def __init__(
        self,
        name: str,
        files: typing.Collection,
        submodules: typing.Collection[T_CodeModuleBase] = tuple(),
    ):
        self._name = name
        self._files = files
        self._submodules = submodules
