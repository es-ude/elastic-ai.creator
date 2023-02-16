from abc import abstractmethod
from typing import Collection, Protocol, TypeVar

from .code_file import CodeFile

T_CodeModule = TypeVar("T_CodeModule", bound="CodeModule")

T_CodeFile = TypeVar("T_CodeFile", bound=CodeFile, covariant=True)


class CodeModule(Protocol[T_CodeFile]):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def files(self) -> Collection[T_CodeFile]:
        ...

    @property
    @abstractmethod
    def submodules(self: T_CodeModule) -> Collection[T_CodeModule]:
        ...
