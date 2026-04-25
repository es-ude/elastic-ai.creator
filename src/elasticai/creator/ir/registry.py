from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from typing import Any, Self, cast, overload

from .datagraph import DataGraph

_registry_types = []


def is_registry(object) -> bool:
    return object in _registry_types


def mark_as_registry[T](object: T) -> T:
    _registry_types.append(object)
    return object


@mark_as_registry
class Registry[G: DataGraph](Mapping[str, G]):
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
        **kwargs,
    ) -> None:
        if items is None:
            self._data: dict[str, G] = dict(**kwargs)
        else:
            if not len(kwargs) == 0:
                raise TypeError(f"unsupported arguments for {type(self)}")
            self._data = dict(items)

    def __getitem__(self, name: str) -> G:
        return cast(G, self._data[name])

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        yield from self._data

    def __or__(self, other: object) -> Self:
        if not isinstance(other, Mapping):
            raise TypeError("unsupported operand")
        return type(self)(**(self._data | dict(other.items())))  # type: ignore

    def add(self, name: str, graph: G) -> Self:
        return type(self)(**(self._data) | {name: graph})

    def apply[G2: DataGraph](self, fn: Callable[[G], G2]) -> "Registry[G2]":
        new_dict = ((k, fn(g)) for k, g in self._data.items())
        return Registry(new_dict)

    def without(self, keys: Collection[str]) -> Self:
        new_dict = ((k, g) for k, g in self._data.items() if k not in keys)
        return type(self)(new_dict)

    def __repr__(self) -> str:
        return f"Registry(**{repr(self._data)})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Registry):
            if set(self.keys()) != other.keys():
                return False
            for k in self.keys():
                if self[k] != other[k]:
                    return False
            return True
        return False
