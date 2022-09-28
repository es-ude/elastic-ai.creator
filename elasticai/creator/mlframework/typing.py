"""Interfaces/Adapters to the used machine learning framework"""

from typing import Any, Callable, Iterable, Iterator, Protocol, runtime_checkable

from numpy.typing import ArrayLike

Index = int | slice | tuple[int | slice]


class Tensor(Protocol):
    def __getitem__(self, index: Index) -> "Tensor":
        ...

    def apply_(self, callable: Callable) -> "Tensor":
        ...

    def detach(self) -> "Tensor":
        ...

    def numpy(self) -> ArrayLike:
        ...


@runtime_checkable
class Parameter(Tensor, Protocol):
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

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        ...

    def state_dict(self) -> dict[str, Any]:
        ...

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        ...
