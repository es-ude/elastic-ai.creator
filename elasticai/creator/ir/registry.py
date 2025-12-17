from collections.abc import Iterator, Mapping
from typing import Self

from .datagraph import DataGraph

_registry_types = []


def is_registry(object) -> bool:
    return object in _registry_types


def mark_as_registry[T](object: T) -> T:
    _registry_types.append(object)
    return object


@mark_as_registry
class Registry[G: DataGraph](Mapping[str, G]):
    def __init__(self, **kwargs: G) -> None:
        self._data: dict[str, G] = dict(**kwargs)

    def __getitem__(self, name: str) -> G:
        return self._data[name]

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        yield from self._data

    def __or__(self, other: object) -> Self:
        if not isinstance(other, Mapping):
            raise TypeError("unsupported operand")
        return type(self)(**(self._data | dict(other.items())))

    def add(self, name: str, graph: G) -> Self:
        return type(self)(**(self._data) | {name: graph})
