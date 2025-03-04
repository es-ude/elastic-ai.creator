import copy
import warnings
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from .graph import Graph
from .name_generation import NameRegistry


class RewriteResult:
    def __init__(
        self,
        *,
        new_graph: Graph[str],
        pattern_to_original: dict[str, str],
        replacement_to_new: dict[str, str],
    ) -> None:
        self.new_graph = new_graph
        self.pattern_to_original = pattern_to_original
        self.replacement_to_new = replacement_to_new


@runtime_checkable
class Matcher(Protocol):
    def __call__(self, *, pattern: Graph, graph: Graph) -> dict[str, str]: ...


@runtime_checkable
class _SeqMatcher(Protocol):
    def __call__(self, *, pattern: Graph, graph: Graph) -> Sequence[dict[str, str]]: ...


class GraphRewriter:
    """Rewrite graphs based on a pattern, interface, and replacement.

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

    :::{note}
    In many cases where you want the matcher to compare graph and pattern attributes, you will have to
    pass a custom matcher and change it before running the rewrite on a new graph.
    E.g. to perform two subsequent rewrites you might have to:
    ```python
    matcher = MyMatcher()
    matcher.set_graph(graph)
    rewriter = GraphRewriter(
        pattern=pattern,
        interface=interface,
        replacement=replacement,
        match=matcher,
        lhs=lhs,
        rhs=rhs,
    )
    result = rewriter.rewrite(graph)
    matcher.set_graph(result.new_graph)
    result = rewriter.rewrite(result.new_graph)
    ```
    :::

    :param pattern: The pattern to match in the graph.
    :param interface: The interface where graph and replacement will be "glued" together.
    :param replacement: The replacement for the pattern.
    :param match: A function that takes a graph and a pattern and returns a dictionary of matches. See [`find_subgraphs`](#elasticai.creator.ir.find_subgraphs) for more information.
    :param lhs: A dict that is used to map the interface to the pattern.
    :param rhs: A dict that is used to map the interface to the replacement.


    """

    def __init__(
        self,
        *,
        pattern: Graph[str],
        interface: Graph[str],
        replacement: Graph[str],
        match: Matcher | _SeqMatcher,
        lhs: Mapping[str, str],
        rhs: Mapping[str, str],
    ) -> None:
        self._pattern = pattern
        self._interface = interface
        self._replacement = replacement
        self._match = match
        self._lhs = lhs
        self._rhs = rhs
        self._rhs_inversed = {self._rhs[node]: node for node in self._interface.nodes}
        self._replacement_nodes_in_interface = self._rhs_inversed.keys()
        if len(self._interface.nodes) != len(self._replacement_nodes_in_interface):
            raise ValueError(
                "Ensure the `rhs` function is injective. The `rhs` function should map each interface node to a unique replacement node."
            )

    def rewrite(self, graph: Graph[str]) -> RewriteResult:
        """Rewrite the graph based on the pattern, interface, and replacement.

        Returns a [`RewriteResult`](#elasticai.creator.ir.graph_rewriting.RewriteResult) object containing the new graph and two dicts mapping nodes from `pattern` to the original graph and nodes from `replacement` to the new graph.
        matches = self._match(graph, self._pattern)
        """
        matches = self._match(pattern=self._pattern, graph=graph)
        if isinstance(matches, dict):
            matches = [matches]
        else:
            warnings.warn(
                "Using a matcher that returns multiple matches. Only the first match will be used. This is deprecated and will be removed in the future.",
                DeprecationWarning,
                stacklevel=2,
            )
        if len(matches) > 0:
            match = matches[0]
        else:
            return RewriteResult(
                new_graph=copy.deepcopy(graph),
                pattern_to_original={},
                replacement_to_new={},
            )
        interface_nodes_in_pattern = set(
            self._lhs[node] for node in self._interface.nodes
        )
        print(match)
        interface_nodes_in_graph = set(
            match[node] for node in interface_nodes_in_pattern
        )

        new_graph: Graph = graph.new()
        nodes_to_be_removed = set(match.values()) - set(interface_nodes_in_graph)
        nodes_to_keep = set(graph.nodes) - nodes_to_be_removed
        name_registry = SingleNewNameGenerator(
            NameRegistry().prepopulate(nodes_to_keep)
        )

        def get_replacement_node_in_new_graph(node: str) -> str:
            if node in self._replacement_nodes_in_interface:
                return match[self._lhs[self._rhs_inversed[node]]]
            if node in nodes_to_keep:
                return name_registry.get_name(node)
            return node

        def get_replacement_edge_for_new_graph(src: str, dst: str) -> tuple[str, str]:
            src = get_replacement_node_in_new_graph(src)
            dst = get_replacement_node_in_new_graph(dst)
            return src, dst

        for node in nodes_to_keep:
            new_graph.add_node(node)

        for node in self._replacement.nodes:
            if node not in self._replacement_nodes_in_interface:
                new_graph.add_node(get_replacement_node_in_new_graph(node))

        for src, dst in graph.iter_edges():
            if src in nodes_to_keep and dst in nodes_to_keep:
                new_graph.add_edge(src, dst)

        for src, dst in self._replacement.iter_edges():
            new_graph.add_edge(*get_replacement_edge_for_new_graph(src, dst))

        return RewriteResult(
            new_graph=new_graph,
            pattern_to_original=match,
            replacement_to_new={
                r: get_replacement_node_in_new_graph(r) for r in self._replacement.nodes
            },
        )


class SingleNewNameGenerator:
    def __init__(self, registry: NameRegistry) -> None:
        self._registry = registry
        self._reversed: dict[str, str] = {}
        self.memory: dict[str, str] = {}

    def get_name(self, name: str) -> str:
        if name in self.memory:
            return self.memory[name]
        new_name = self._registry.get_unique_name(name)
        self.memory[name] = new_name
        self._reversed[new_name] = name
        return new_name
