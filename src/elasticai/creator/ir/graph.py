from abc import abstractmethod
from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import Any, Protocol, Self, cast


class ReadOnlyGraph[N: Hashable, E](Protocol):
    @property
    @abstractmethod
    def successors(self) -> Mapping[N, Mapping[N, E]]: ...

    @property
    @abstractmethod
    def predecessors(self) -> Mapping[N, Mapping[N, E]]: ...


class Graph[N: Hashable, E](ReadOnlyGraph[N, E], Protocol):
    @abstractmethod
    def add_node(self, node: N, /) -> Self: ...

    @abstractmethod
    def add_edge(self, src: N, dst: N, attributes: E | None = None, /) -> Self: ...

    @abstractmethod
    def remove_node(self, node: N, /) -> Self: ...

    @abstractmethod
    def remove_edge(self, src: N, dst: N, /) -> Self: ...

    @abstractmethod
    def add_nodes(self, *nodes: N) -> Self: ...

    @abstractmethod
    def add_edges(self, *edges: tuple[N, N, E] | tuple[N, N]) -> Self: ...


class AdjacencyMap[K, V](Mapping[K, Mapping[K, V]]):
    def __init__(self, mapping: dict[K, dict[K, V]] | None = None) -> None:
        self._mapping = mapping or {}

    def __getitem__(self, key: K) -> Mapping[K, V]:
        return self._mapping[key].keys().mapping

    def __contains__(self, key: object) -> bool:
        return key in self._mapping

    def __iter__(self) -> Iterator[K]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __repr__(self) -> str:
        return f"AdjacencyMap({repr(self._mapping)})"

    def join(self, other: Mapping[K, Mapping[K, V]] | dict[K, dict[K, V]]) -> Self:
        if len(other) == 0:
            return self
        new_dict: dict[K, dict[K, V]] = {}
        joined_keys_from_other = set()
        for key in self._mapping:
            new_dict[key] = self._mapping[key]

            if key in other:
                new_dict[key] = new_dict[key] | dict(other[key].items())
                joined_keys_from_other.add(key)
        for key in other:
            if key not in joined_keys_from_other:
                new_dict[key] = dict(other[key].items())
        return cast(Self, AdjacencyMap(new_dict))

    def drop(self, k1: K, k2: K | None = None) -> Self:
        def remove_key(d: dict[K, V], key: K) -> dict[K, V]:
            return {k: v for k, v in d.items() if k != key}

        if k2 is None:
            new_dict: dict[K, dict[K, V]] = {
                k: remove_key(v, k1) for k, v in self._mapping.items() if k != k1
            }
            return type(self)(new_dict)
        else:
            new_mapping = self._mapping | {
                k1: {k: v for k, v in self._mapping[k1].items() if k != k2}
            }
            return type(self)(new_mapping)


class GraphImpl[T: Hashable, E](Graph[T, E]):
    def __init__(
        self,
        default_edge_attributes_factory: Callable[[], E],
        predecessors: AdjacencyMap[T, E] = AdjacencyMap(),
        successors: AdjacencyMap[T, E] = AdjacencyMap(),
    ) -> None:
        self._predecessors = predecessors
        self._successors = successors
        self._default_edge_attributes_factory = default_edge_attributes_factory

    def add_node(self, node: T, /) -> Self:
        return self.add_nodes(node)

    def add_nodes(self, *nodes: T) -> Self:
        additional_predecessors: dict[T, dict[T, E]] = {}
        additional_successors: dict[T, dict[T, E]] = {}
        for node in nodes:
            if node not in self.predecessors:
                additional_successors[node] = {}
                additional_predecessors[node] = {}
        new_predecessors = self.predecessors.join(additional_predecessors)
        new_successors = self.successors.join(additional_successors)
        return type(self)(
            self._default_edge_attributes_factory,
            predecessors=new_predecessors,
            successors=new_successors,
        )

    @property
    def successors(self) -> AdjacencyMap[T, E]:
        return self._successors

    @property
    def predecessors(self) -> AdjacencyMap[T, E]:
        return self._predecessors

    def add_edge(self, src: T, dst: T, attributes: E | None = None, /) -> Self:
        return self.add_edges((src, dst, attributes))

    def __eq__(self, other: object) -> bool:
        if hasattr(other, "successors"):
            return self.successors == other.successors
        return False

    def add_edges(self, *edges: tuple[T, T, E | None] | tuple[T, T]) -> Self:
        def dispatch_arg(edge: tuple[T, T, E | None] | tuple[T, T]) -> tuple[T, T, E]:
            if len(edge) == 3:
                src, dst, attributes = edge
            elif len(edge) == 2:
                src, dst = edge
                attributes = None
            else:
                raise TypeError(f"Invalid edge argument {edge}.")
            if attributes is None:
                return src, dst, self._default_edge_attributes_factory()
            else:
                return src, dst, attributes

        _edges = tuple(map(dispatch_arg, edges))

        additional_successors: dict[T, dict[T, Any]] = {
            src: {dst: attributes or self._default_edge_attributes_factory()}
            for src, dst, attributes in _edges
        }
        additional_predecessors: dict[T, dict[T, Any]] = {
            dst: {src: attributes or self._default_edge_attributes_factory()}
            for src, dst, attributes in _edges
        }
        for src, dst, _ in _edges:
            if dst not in self.successors and dst not in additional_successors:
                additional_successors[dst] = {}
            if src not in self.predecessors and src not in additional_predecessors:
                additional_predecessors[src] = {}
        new_successors = self.successors.join(additional_successors)
        new_predecessors = self.predecessors.join(additional_predecessors)
        return type(self)(
            self._default_edge_attributes_factory,
            predecessors=new_predecessors,
            successors=new_successors,
        )

    def remove_edge(self, src: T, dst: T, /) -> Self:
        new_predecessors = self.predecessors.drop(dst, src)
        new_successors = self.successors.drop(src, dst)
        return type(self)(
            self._default_edge_attributes_factory,
            predecessors=new_predecessors,
            successors=new_successors,
        )

    def remove_node(self, node: T, /) -> Self:
        new_predecessors = self.predecessors.drop(node)
        new_successors = self.successors.drop(node)
        return type(self)(
            self._default_edge_attributes_factory,
            predecessors=new_predecessors,
            successors=new_successors,
        )
