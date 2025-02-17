import copy
from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

from .core import Edge, Node
from .graph import Graph
from .name_generation import NameRegistry

EdgeT = TypeVar("EdgeT", bound=Edge)

PNodeT = TypeVar("PNodeT", bound=Node)
INodeT = TypeVar("INodeT", bound=Node)
RNodeT = TypeVar("RNodeT", bound=Node)
GNodeT = TypeVar("GNodeT", bound=Node)


class GraphRewriter(Generic[PNodeT, INodeT, RNodeT, GNodeT, EdgeT]):
    """Rewrite graphs based on a pattern, interface, and replacement.

    :param: `pattern`: The pattern to match in the graph.
    :param: `interface`: The interface where graph and replacement will be "glued" together.
    :param: `replacement`: The replacement for the pattern.
    :param: `match`: A function that takes a graph and a pattern and returns a dictionary of matches. See [`find_subgraphs`](#elasticai.creator.ir.find_subgraphs) for more information.
    :param: `lhs`: A function that is used to map the interface to the pattern.
    :param: `rhs`: A function that is used to map the interface to the replacement.

    The terminology here is based on the double pushout approach to graph rewriting.
    The algorithm will find the first occurence of pattern in the graph using the `match` function.
    Then, it will create a new graph by replacing the pattern with the replacement.
    The structures specified by `interface` are excluded from this replacement.
    Instead the interface is used as the "glue" between the original graph and the replacement.
    The functions `lhs` and `rhs` are used to identify the `interface` nodes in the `pattern` and `replacement` respectively.

    :::{important}
    * The `rhs` function is required to be injective. This means that each interface node should be mapped to a unique replacement node.
    * The algorithm will stop on the first match. If you need to replace multiple matches you have to call the `rewrite` function multiple times.
    * The nodes from `replacement` will be automatically renamed to avoid conflicts with the nodes in the original graph, i.e., if node `a` exists already we will add a new node `a_1`.
    :::



    """

    def __init__(
        self,
        pattern: Graph[PNodeT, EdgeT],
        interface: Graph[INodeT, EdgeT],
        replacement: Graph[RNodeT, EdgeT],
        match: Callable[
            [Graph[GNodeT, EdgeT], Graph[PNodeT, EdgeT]], Sequence[dict[str, str]]
        ],
        lhs: Callable[[str], str],
        rhs: Callable[[str], str],
    ) -> None:
        self._pattern = pattern
        self._interface = interface
        self._replacement = replacement
        self._match = match
        self._lhs = lhs
        self._rhs = rhs
        self._rhs_inversed = {self._rhs(node): node for node in self._interface.nodes}
        self._interface_nodes_in_replacement = set(self._rhs_inversed.keys())
        if len(self._interface.nodes) != len(self._interface_nodes_in_replacement):
            raise ValueError(
                "Ensure the `rhs` function is injective. The `rhs` function should map each interface node to a unique replacement node."
            )

    def rewrite(self, graph: Graph[GNodeT, EdgeT]) -> Graph[GNodeT, EdgeT]:
        matches = self._match(graph, self._pattern)
        if len(matches) > 0:
            match = matches[0]
        else:
            return copy.deepcopy(graph)
        interface_nodes_in_pattern = set(
            self._lhs(node) for node in self._interface.nodes
        )
        interface_nodes_in_graph = set(
            match[node] for node in interface_nodes_in_pattern
        )

        new_graph = graph.get_empty_copy()
        nodes_to_be_removed = set(match.values()) - set(interface_nodes_in_graph)
        nodes_to_keep = set(graph.nodes.keys()) - nodes_to_be_removed
        name_registry = SingleNewNameGenerator(
            NameRegistry().prepopulate(nodes_to_keep)
        )

        def get_name_for_replacement_node_in_new_graph(node: str) -> str:
            if node in self._interface_nodes_in_replacement:
                return match[self._lhs(self._rhs_inversed[node])]
            if node in nodes_to_keep:
                return name_registry.get_name(node)
            return node

        def get_name_for_old_node_in_new_graph(node: str) -> str:
            return node

        def get_old_node_for_new_graph(node: str) -> Node:
            node_data = copy.deepcopy(graph.nodes[node].data)
            node_data["name"] = get_name_for_old_node_in_new_graph(node)
            return Node(node_data)

        def get_replacement_node_for_new_graph(node: str) -> Node:
            node_data = copy.deepcopy(self._replacement.nodes[node].data)
            node_data["name"] = get_name_for_replacement_node_in_new_graph(node)
            return Node(node_data)

        def get_old_edge_for_new_graph(src: str, sink: str) -> Edge:
            old_edge = graph.edges[(src, sink)]
            edge_data = copy.deepcopy(old_edge.data)
            edge_data["src"] = get_name_for_old_node_in_new_graph(old_edge.src)
            edge_data["sink"] = get_name_for_old_node_in_new_graph(old_edge.sink)
            return Edge(edge_data)

        def get_replacement_edge_for_new_graph(src: str, sink: str) -> Edge:
            replacement_edge = self._replacement.edges[(src, sink)]
            edge_data = copy.deepcopy(replacement_edge.data)
            edge_data["src"] = get_name_for_replacement_node_in_new_graph(
                replacement_edge.src
            )
            edge_data["sink"] = get_name_for_replacement_node_in_new_graph(
                replacement_edge.sink
            )
            return Edge(edge_data)

        for node in nodes_to_keep:
            new_graph.add_node(get_old_node_for_new_graph(node))

        for node in self._replacement.nodes:
            if node not in self._interface_nodes_in_replacement:
                new_graph.add_node(get_replacement_node_for_new_graph(node))

        for src, sink in graph.edges:
            if src in nodes_to_keep and sink in nodes_to_keep:
                new_graph.add_edge(get_old_edge_for_new_graph(src, sink))

        for src, sink in self._replacement.edges:
            new_graph.add_edge(get_replacement_edge_for_new_graph(src, sink))
        return new_graph


class SingleNewNameGenerator:
    def __init__(self, registry: NameRegistry) -> None:
        self._registry = registry
        self._reversed: dict[str, str] = {}
        self._memory: dict[str, str] = {}

    def get_name(self, name: str) -> str:
        if name in self._memory:
            return self._memory[name]
        new_name = self._registry.get_unique_name(name)
        self._memory[name] = new_name
        self._reversed[new_name] = name
        return new_name
