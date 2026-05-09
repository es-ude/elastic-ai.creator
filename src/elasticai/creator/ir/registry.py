from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from typing import Self, overload, override

from .datagraph import DataGraph, Edge, Node

_registry_types: list[type] = []


def is_registry(obj: type) -> bool:
    return obj in _registry_types


def mark_as_registry[T](object: type[T]) -> type[T]:
    _registry_types.append(object)
    return object


@mark_as_registry
class Registry[G: DataGraph[Node, Edge]](Mapping[str, G]):
    @overload
    def __init__(self, items: Iterable[tuple[str, G]], /) -> None: ...
    @overload
    def __init__(self, items: Mapping[str, G], /) -> None: ...
    @overload
    def __init__(self, /, **kwargs: G) -> None: ...

    def __init__(
        self,
        items: Iterable[tuple[str, G]] | Mapping[str, G] | None = None,
        /,
        **kwargs: G,
    ) -> None:  # zuban: ignore[misc]
        if items is None:
            self._data: dict[str, G] = kwargs
        else:
            if not len(kwargs) == 0:
                raise TypeError(f"unsupported arguments for {type(self)}")
            self._data = dict(items)  # zuban: ignore[call-overload]

    @override
    def __getitem__(self, name: str) -> G:
        return self._data[name]

    @override
    def __len__(self) -> int:
        return len(self._data)

    @override
    def __contains__(self, key: object) -> bool:
        return key in self._data

    @override
    def __iter__(self) -> Iterator[str]:
        yield from self._data

    def __or__(self, other: Mapping[str, G]) -> Self:
        return type(self)((self._data | dict(other.items())))

    def add(self, name: str, graph: G) -> Self:
        return type(self)(**(self._data) | {name: graph})

    def apply[G2: DataGraph[Node, Edge]](self, fn: Callable[[G], G2]) -> "Registry[G2]":
        new_dict = ((k, fn(g)) for k, g in self._data.items())
        return Registry(new_dict)

    def without(self, keys: Collection[str]) -> Self:
        new_dict = ((k, g) for k, g in self._data.items() if k not in keys)
        return type(self)(new_dict)

    @override
    def __repr__(self) -> str:
        return f"Registry(**{repr(self._data)})"

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Registry):
            if self.keys() != other.keys():
                return False
            for k in self.keys():
                if self[k] != other[k]:
                    return False
            return True
        return False
