from abc import abstractmethod
from typing import Protocol, Iterable, Union, Callable

Indices = tuple[Union[int, slice], ...]


class Tensor(Protocol):
    @abstractmethod
    def __getitem__(self, index: Indices) -> "Tensor":
        ...

    @abstractmethod
    def as_strided(
        self, size: Union[tuple[int, ...], int], stride: Union[tuple[int, ...], int]
    ) -> "Tensor":
        ...

    @abstractmethod
    def select(self, dim: int, index: int) -> "Tensor":
        ...

    @property
    @abstractmethod
    def shape(self):
        ...


class Module(Protocol):
    @property
    @abstractmethod
    def training(self) -> bool:
        pass

    @abstractmethod
    def extra_repr(self) -> str:
        pass

    @abstractmethod
    def named_children(self) -> Iterable[tuple[str, "Module"]]:
        pass

    @abstractmethod
    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        pass


TensorLike = Union[Callable[[], "TensorLike"], Tensor]
