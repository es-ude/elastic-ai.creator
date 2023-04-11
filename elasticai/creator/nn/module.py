from abc import abstractmethod
from typing import Protocol, TypeVar

from elasticai.creator.hdl.translatable import Savable

T = TypeVar("T", bound=Savable, covariant=True)


class Tensor(Protocol):
    ...


class Module(Protocol[T]):
    def __call__(self, x) -> Tensor:
        ...

    @abstractmethod
    def translate(self) -> T:
        ...
