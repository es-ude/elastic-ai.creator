from abc import abstractmethod
from typing import Protocol, TypeVar

T_contra = TypeVar("T_contra", contravariant=True)


class Acceptor(Protocol[T_contra]):
    @abstractmethod
    def accepts(self, source: T_contra) -> bool:
        ...
