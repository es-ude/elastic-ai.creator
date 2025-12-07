from abc import abstractmethod
from collections.abc import Callable, Collection, Hashable, Iterable, Iterator, Mapping
from typing import Any, Protocol, Self

from elasticai.creator.graph import (
    Graph,
    find_all_subgraphs,
    get_rewriteable_matches,
    rewrite,
)


class Node(Protocol):
    @property
    @abstractmethod
    def data(self) -> dict[str, Any]: ...

    @data.setter
    @abstractmethod
    def data(self, data: dict[str, Any]) -> None: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class Edge(Protocol):
    @property
    @abstractmethod
    def data(self) -> dict[str, Any]: ...

    @data.setter
    @abstractmethod
    def data(self, data: dict[str, Any]) -> None: ...
    @property
    @abstractmethod
    def src(self) -> str: ...

    @property
    @abstractmethod
    def dst(self) -> str: ...


class ReadOnlyDataGraph[N: Node, E: Edge](Protocol):
    @property
    def graph(self) -> Graph[str]: ...

    @property
    @abstractmethod
    def nodes(self) -> Mapping[str, N]: ...

    @property
    @abstractmethod
    def edges(self) -> Mapping[tuple[str, str], E]: ...

    @abstractmethod
    def successors(self, node: str) -> Mapping[str, N]: ...

    @abstractmethod
    def predecessors(self, node: str) -> Mapping[str, N]: ...

    @property
    @abstractmethod
    def data(self) -> dict[str, Any]: ...


class DataGraph[N: Node, E: Edge](ReadOnlyDataGraph[N, E], Protocol):
    @abstractmethod
    def add_node(self, node: N) -> Self: ...

    @abstractmethod
    def add_edge(self, edge: E) -> Self: ...

    @property
    @abstractmethod
    def data(self) -> dict[str, Any]: ...

    @data.setter
    @abstractmethod
    def data(self, data: dict[str, Any]) -> None: ...


class RemappedSubImplementation[N: Node, E: Edge](ReadOnlyDataGraph[N, E]):
    """Use a given mapping to make the data dictionary accessible via node names from graph.

    This allows to access the data dictionary using the names from the graph.
    The provided mapping is used to map the node names of the graph to the node names in the data dictionary.
    This is used e.g., to initialize new nodes, that were replaced during rewriting.
    The mapping is assumed to be one-to-one, i.e. each node name in the graph maps to exactly one node name in the data dictionary.

    """

    def __init__(
        self,
        mapping: dict[str, str],
        graph: Graph[str],
        data: dict[str, Any],
        node_fn: Callable[[str, dict[str, Any]], N],
        edge_fn: Callable[[str, str, dict[str, Any]], E],
    ):
        self._mapping = mapping
        self._inverted = dict(((v, k) for k, v in mapping.items()))
        self._data = data
        self._graph = graph
        self._node_constr = node_fn
        self._edge_constr = lambda src_dst, d: edge_fn(src_dst[0], src_dst[1], d)

    @property
    def graph(self) -> Graph[str]:
        return self._graph

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def _original_name(self, node: str):
        return self._mapping[node]

    def _mapped_name(self, node: str):
        return self._inverted[node]

    @property
    def nodes(self) -> Mapping[str, N]:
        return self._create_remapped_nodes(self._graph.nodes)

    @property
    def edges(self) -> Mapping[tuple[str, str], E]:
        remapping = {
            (self._original_name(src), self._original_name(dst)): (src, dst)
            for src, dst in self._graph.iter_edges()
        }
        return _RemappedData(
            remapping=remapping,
            data=self._data["edges"],
            iterable=self._graph.iter_edges(),
            factory_fn=self._edge_constr,
        )

    def successors(self, node: str) -> Mapping[str, N]:
        return self._create_remapped_nodes(
            self._graph.successors[self._inverted[node]],
        )

    def _create_remapped_nodes(self, iterable: Iterable[str]) -> Mapping[str, N]:
        return _RemappedData(
            self._mapping, self._data, iterable, factory_fn=self._node_constr
        )

    def predecessors(self, node: str) -> Mapping[str, N]:
        return self._create_remapped_nodes(
            self._graph.predecessors[self._inverted[node]]
        )


class _RemappedData[K: Hashable, D](Mapping[K, D]):
    def __init__(
        self,
        remapping: dict[K, K],
        data: dict[K, Any],
        iterable: Iterable[K],
        factory_fn: Callable[[K, dict[str, Any]], D],
    ) -> None:
        self._remapping = remapping
        self._data = data
        self._iter = iterable
        self._create_item = factory_fn

    def __getitem__(self, key) -> D:
        return self._create_item(key, self._data[self._remapping[key]])

    def __len__(self) -> int:
        return len(self._remapping)

    def __iter__(self) -> Iterator[Any]:
        for key in iter(self._iter):
            if key in self:
                yield key

    def __contains__(self, key: Any) -> bool:
        return key in self._remapping


type Rule = Callable[[DataGraph], DataGraph]


class CompositeRule:
    def __init__(self, rules: Iterable[Rule]) -> None:
        self._rules = list(rules)

    def __call__(self, impl: DataGraph) -> DataGraph:
        for rule in self._rules:
            impl = rule(impl)
        return impl


class Pattern(Protocol):
    @property
    @abstractmethod
    def graph(self) -> DataGraph: ...

    @property
    def interface(self) -> Collection[str]: ...

    @abstractmethod
    def match(self, g: ReadOnlyDataGraph) -> list[dict[str, str]]: ...


class StdPattern(Pattern):
    def __init__(
        self,
        graph: DataGraph,
        node_constraint: Callable[[Node, Node], bool],
        interface: Collection[str],
    ):
        self._pattern = graph
        self._node_constraint = node_constraint
        self._interface = interface

    @property
    def graph(self) -> DataGraph:
        return self._pattern

    @property
    def interface(self) -> Collection[str]:
        return self._interface

    def match(self, g: ReadOnlyDataGraph) -> list[dict[str, str]]:
        matches = find_all_subgraphs(
            graph=g.graph,
            pattern=self.graph.graph,
            node_constraint=lambda p_node, g_node: self._node_constraint(
                self._pattern.nodes[p_node],
                g.nodes[g_node],
            ),
        )
        matches = list(
            get_rewriteable_matches(
                matches=matches, original=g.graph, interface_nodes=set(self._interface)
            )
        )
        return matches


class IrFactory[N: Node, E: Edge, G: DataGraph](Protocol):
    @abstractmethod
    def node(self, name: str, data: Mapping[str, Any]) -> N: ...

    @abstractmethod
    def edge(self, src: str, dst: str, data: Mapping[str, Any]) -> E: ...

    @abstractmethod
    def data_graph(self) -> G: ...


class PatternRule[N: Node, E: Edge, G: DataGraph]:
    def __init__(self, spec: "PatternRuleSpec", ir_factory: IrFactory[N, E, G]):
        self._spec = spec
        self._ir_factory = ir_factory

    def __call__(self, graph: DataGraph) -> DataGraph:
        return self.apply(graph)

    def apply(self, graph: ReadOnlyDataGraph) -> DataGraph:
        self._validate_pattern(self._spec.pattern)
        matches = list(self._spec.pattern.match(graph))

        matched_subgraphs = [
            RemappedSubImplementation[N, E](
                mapping=match,
                graph=self._spec.pattern.graph.graph,
                data=graph.data["nodes"],
                node_fn=self._ir_factory.node,
                edge_fn=self._ir_factory.edge,
            )
            for match in matches
        ]
        replacements = [
            self._spec.create_replacement(matched) for matched in matched_subgraphs
        ]
        self._validate_replacements(replacements)
        new_graph, replacement_maps = self._rewrite_raw_graph_for_matches(
            graph, matches, replacements
        )
        return self._build_data_graph(
            new_raw_graph=new_graph,
            original=graph,
            replacement_map=replacement_maps,
        )

    def _build_data_graph(
        self,
        new_raw_graph: Graph[str],
        original: ReadOnlyDataGraph,
        replacement_map: dict[str, tuple[str, ReadOnlyDataGraph]],
    ) -> DataGraph:
        impl = self._ir_factory.data_graph()

        combined_map = replacement_map | {
            name: (name, original)
            for name in original.nodes
            if name in new_raw_graph.nodes and name
        }

        def copy_node(name):
            old_name, graph = combined_map[name]
            impl.add_node(
                self._ir_factory.node(name=name, data=graph.nodes[old_name].data)
            )

        def edge_lies_within_replacement(src_dst):
            src, dst = src_dst
            if src in replacement_map and dst in replacement_map:
                return True
            return False

        def copy_edge(src_dst):
            src, dst = src_dst
            if edge_lies_within_replacement(src_dst):
                old_src_name, graph = replacement_map[src]
                old_dst_name, _ = replacement_map[dst]
            else:
                old_src_name, graph = src, original
                old_dst_name, _ = dst, original

            impl.add_edge(
                self._ir_factory.edge(
                    src=src,
                    dst=dst,
                    data=graph.edges[(old_src_name, old_dst_name)].data,
                )
            )

        for node_name in new_raw_graph.nodes:
            copy_node(node_name)

        for src, dst in new_raw_graph.iter_edges():
            copy_edge((src, dst))

        return impl

    def _validate_replacements(self, replacements: list[ReadOnlyDataGraph]) -> None:
        for replacement in replacements:
            missing_interface_nodes = set(self._spec.interface) - set(
                replacement.nodes.keys()
            )
            if missing_interface_nodes:
                raise ValueError(
                    f"Replacement is missing interface nodes: {missing_interface_nodes}"
                )

    def _validate_pattern(self, pattern: Pattern) -> None:
        missing_interface_nodes = set(self._spec.interface) - set(
            pattern.graph.nodes.keys()
        )
        if missing_interface_nodes:
            raise ValueError(
                f"Pattern Graph is missing interface nodes: {missing_interface_nodes}"
            )

    def _rewrite_raw_graph_for_matches(
        self,
        graph: ReadOnlyDataGraph,
        matches: list[dict[str, str]],
        replacements: list[ReadOnlyDataGraph],
    ) -> tuple[Graph[str], dict[str, tuple[str, ReadOnlyDataGraph]]]:
        rewritten = graph.graph
        full_replacement_map = {}
        for match, replacement in zip(matches, replacements):
            rewritten, replacement_map = rewrite(
                replacement=replacement.graph,
                original=rewritten,
                match=match,
                lhs={x: x for x in self._spec.interface},
                rhs={x: x for x in self._spec.interface},
            )
            for replacement_node_name, new_node_name in replacement_map.items():
                full_replacement_map[new_node_name] = (
                    replacement_node_name,
                    replacement,
                )

        return rewritten, full_replacement_map


class PatternRuleSpec:
    """Specifies how `PatternRule` should create a new `DataGraph` from an existing one.

    It consists of

    - a pattern to match in the original DataGraph.
    - a replacement to apply when the pattern is matched. The
    `PatternRule` will use the `replacement_fn` function to create a
    new `DataGraph`. The function receives the matched
    part of the DataGraph as an argument, so you can build
    the replacement based on the parameters found in your
    matched pattern. The node names in this matched
    subimplementation are remapped from the original graph to
    the pattern graph, so you can access the data dictionary
    using the names from the pattern graph. E.g., if the
    pattern specifies a node `'conv'` you can acces the data of
    the original DataGraph that this node corresponds to
    using `matched.nodes['conv'].data`.
    - an interface that specifies which nodes are part of the
    pattern and of the replacement, nodes that are part of the
    interface are neither replaced nor initialized.
    Instead they are used to connect the pattern and the
    replacement.
    """

    def __init__(
        self,
        pattern: Pattern,
        replacement_fn: Callable[[ReadOnlyDataGraph], ReadOnlyDataGraph],
    ):
        self.pattern = pattern
        self.create_replacement = replacement_fn

    @property
    def interface(self) -> Collection[str]:
        return self.pattern.interface
