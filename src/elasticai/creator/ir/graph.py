from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from typing import Protocol, Self, cast, override, runtime_checkable


@runtime_checkable
class ReadOnlyGraph[N: Hashable, E](Protocol):
    @property
    def successors(
        self,
    ) -> Mapping[N, Mapping[N, E]]: ...

    @property
    def predecessors(self) -> Mapping[N, Mapping[N, E]]: ...


class Graph[N: Hashable, E](ReadOnlyGraph[N, E], Protocol):
    def add_node(self, node: N, /) -> Self: ...

    def add_edge(self, src: N, dst: N, attributes: E | None = None, /) -> Self: ...

    def remove_node(self, node: N, /) -> Self: ...

    def remove_edge(self, src: N, dst: N, /) -> Self: ...

    def add_nodes(self, *nodes: N) -> Self: ...

    def add_edges(self, *edges: tuple[N, N, E] | tuple[N, N]) -> Self: ...


class AdjacencyMap[K, V](Mapping[K, Mapping[K, V]]):
    def __init__(self, mapping: dict[K, dict[K, V]] | None = None) -> None:
        self._mapping: dict[K, dict[K, V]] = mapping or {}

    @override
    def __getitem__(self, key: K) -> Mapping[K, V]:
        return self._mapping[key].keys().mapping

    @override
    def __contains__(self, key: object) -> bool:
        return key in self._mapping

    @override
    def __iter__(self) -> Iterator[K]:
        return iter(self._mapping)

    @override
    def __len__(self) -> int:
        return len(self._mapping)

    @override
    def __repr__(self) -> str:
        return f"AdjacencyMap({repr(self._mapping)})"

    def join(self, other: Mapping[K, Mapping[K, V]] | dict[K, dict[K, V]]) -> Self:
        if len(other) == 0:
            return self
        new_dict: dict[K, dict[K, V]] = {}
        joined_keys_from_other: set[K] = set()
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
        predecessors: AdjacencyMap[T, E] = AdjacencyMap(),  # pyright: ignore[reportCallInDefaultInitializer]
        successors: AdjacencyMap[T, E] = AdjacencyMap(),  # pyright: ignore[reportCallInDefaultInitializer]  these are ok because we know that map is immutable
    ) -> None:
        self._predecessors: AdjacencyMap[T, E] = predecessors
        self._successors: AdjacencyMap[T, E] = successors
        self._default_edge_attributes_factory: Callable[[], E] = (
            default_edge_attributes_factory
        )

    @override
    def add_node(self, node: T, /) -> Self:
        return self.add_nodes(node)

    @override
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
    @override
    def successors(self) -> AdjacencyMap[T, E]:
        return self._successors

    @property
    @override
    def predecessors(self) -> AdjacencyMap[T, E]:
        return self._predecessors

    @override
    def add_edge(self, src: T, dst: T, attributes: E | None = None, /) -> Self:
        return self.add_edges((src, dst, attributes))

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, ReadOnlyGraph):
            return self.successors == other.successors
        return False

    @override
    def add_edges(self, *edges: tuple[T, T, E | None] | tuple[T, T]) -> Self:
        def dispatch_arg(edge: tuple[T, T, E | None] | tuple[T, T]) -> tuple[T, T, E]:
            match edge:
                case (src, dst):
                    return src, dst, self._default_edge_attributes_factory()
                case (src, dst, None):
                    return src, dst, self._default_edge_attributes_factory()
                case (src, dst, attributes):
                    return src, dst, attributes  # zuban: ignore[return-value]
            raise TypeError(f"Invalid edge argument {edge}.")

        _edges = tuple(map(dispatch_arg, edges))

        def reverse_src_dst(
            edges: Iterable[tuple[T, T, E]],
        ) -> Iterator[tuple[T, T, E]]:
            for a, b, attributes in edges:
                yield b, a, attributes

        def build_edge_dict(edges: Iterable[tuple[T, T, E]]) -> dict[T, dict[T, E]]:
            d: dict[T, dict[T, E]] = {}
            for a, b, attributes in edges:
                if a not in d:
                    d[a] = {}
                d[a][b] = attributes or self._default_edge_attributes_factory()
            return d

        additional_successors = build_edge_dict(_edges)
        additional_predecessors = build_edge_dict(reverse_src_dst(_edges))
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

    @override
    def remove_edge(self, src: T, dst: T, /) -> Self:
        new_predecessors = self.predecessors.drop(dst, src)
        new_successors = self.successors.drop(src, dst)
        return type(self)(
            self._default_edge_attributes_factory,
            predecessors=new_predecessors,
            successors=new_successors,
        )

    @override
    def remove_node(self, node: T, /) -> Self:
        new_predecessors = self.predecessors.drop(node)
        new_successors = self.successors.drop(node)
        return type(self)(
            self._default_edge_attributes_factory,
            predecessors=new_predecessors,
            successors=new_successors,
        )
