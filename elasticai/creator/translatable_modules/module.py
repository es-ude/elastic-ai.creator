from abc import abstractmethod
from typing import Protocol, TypeVar

from elasticai.creator.hdl.translatable import Saveable

T = TypeVar("T", bound=Saveable, covariant=True)


class Tensor(Protocol):
    ...


class Module(Protocol[T]):
    def __call__(self, x) -> Tensor:
        ...

    def eval(self) -> None:
        ...

    def train(self) -> None:
        ...

    @abstractmethod
    def translate(self) -> T:
        ...
