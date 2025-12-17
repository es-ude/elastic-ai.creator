from collections.abc import Hashable, Iterable, Mapping
from typing import Protocol


class Graph[T: Hashable](Protocol):
    @property
    def successors(self) -> Mapping[T, Iterable[T]]: ...

    @property
    def predecessors(self) -> Mapping[T, Iterable[T]]: ...
