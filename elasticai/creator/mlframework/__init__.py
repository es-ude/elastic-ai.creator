from typing import Iterable, Protocol, Union, runtime_checkable

from numpy.typing import ArrayLike

Index = Union[int, slice, tuple[Union[int, slice]]]


class Tensor(Protocol):
    def __getitem__(self, index: Index) -> "Tensor":
        ...

    def detach(self) -> "Tensor":
        ...

    def numpy(self) -> ArrayLike:
        ...


@runtime_checkable
class Module(Protocol):
    @property
    def training(self) -> bool:
        ...

    def extra_repr(self) -> str:
        ...

    def named_children(self) -> Iterable[tuple[str, "Module"]]:
        ...

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        ...
