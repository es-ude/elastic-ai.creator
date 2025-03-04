import copy
import warnings
from collections.abc import Mapping, Sequence
from typing import Hashable, Protocol, TypeVar, runtime_checkable

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


TP = TypeVar("TP", bound=Hashable)
TG = TypeVar("TG", bound=Hashable)
TR = TypeVar("TR", bound=Hashable)
TI = TypeVar("TI", bound=Hashable)


def rewrite(
    *,
    interface: Graph[str],
    replacement: Graph[str],
    original: Graph[str],
    match: Mapping[str, str],
    lhs: Mapping[str, str],
    rhs: Mapping[str, str],
) -> tuple[Graph[str], dict[str, str]]:
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
    - The nodes from the replacement graph that aren't part of the interface will be
        automatically renamed to avoid conflicts with nodes in the original graph.
    - Interface nodes serve as connection points between the original graph and replacement.
    - In many cases where you want the matcher to compare graph and pattern attributes, you
        will have to pass a custom matcher and change it before running the rewrite on a new
        graph. E.g. to perform two subsequent rewrites you might have to:
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


    :param interface: The interface graph that defines which nodes should be preserved during rewriting.
    :param replacement: The graph that will replace the matched pattern in the original graph.
    :param graph: The original graph where the pattern will be replaced.
    :param match: A dictionary mapping nodes from pattern to nodes in the original graph.
    :param lhs: Maps interface nodes to their corresponding nodes in the pattern (Left Hand Side).
    :param rhs: Maps interface nodes to their corresponding nodes in the replacement (Right Hand Side).
    :return: A tuple containing the new graph after the rewrite operation and a dictionary mapping replacement node names to new node names.
    :raises ValueError: If the `rhs` function is not injective (when different interface nodes map to the same replacement node).
    """
    rhs_inversed = {rhs[node]: node for node in interface.nodes}
    replacement_nodes_in_interface = set(rhs_inversed.keys())
    if len(interface.nodes) != len(replacement_nodes_in_interface):
        raise ValueError(
            "Ensure the `rhs` function is injective. The `rhs` function should map each interface node to a unique replacement node."
        )
    interface_nodes_in_pattern = set(lhs[node] for node in interface.nodes)
    interface_nodes_in_graph = set(match[node] for node in interface_nodes_in_pattern)

    new_graph: Graph = original.new()
    nodes_to_be_removed = set(match.values()) - set(interface_nodes_in_graph)
    nodes_to_keep = set(original.nodes) - nodes_to_be_removed
    name_registry = SingleNewNameGenerator(NameRegistry().prepopulate(nodes_to_keep))
    new_replacement_names: dict[str, str] = {}

    for node in set(replacement.nodes):
        if node in replacement_nodes_in_interface:
            new_replacement_names[node] = match[lhs[rhs_inversed[node]]]
        else:
            new_replacement_names[node] = name_registry.get_name(node)

    def get_replacement_node_in_new_graph(node: str) -> str:
        if node in replacement_nodes_in_interface:
            return match[lhs[rhs_inversed[node]]]
        return new_replacement_names[node]

    def get_replacement_edge_for_new_graph(src: str, dst: str) -> tuple[str, str]:
        src = get_replacement_node_in_new_graph(src)
        dst = get_replacement_node_in_new_graph(dst)
        return src, dst

    for node in nodes_to_keep:
        new_graph.add_node(node)

    for node in replacement.nodes:
        if node not in replacement_nodes_in_interface:
            new_graph.add_node(get_replacement_node_in_new_graph(node))

    for src, dst in original.iter_edges():
        if src in nodes_to_keep and dst in nodes_to_keep:
            new_graph.add_edge(src, dst)

    for src, dst in replacement.iter_edges():
        new_graph.add_edge(*get_replacement_edge_for_new_graph(src, dst))

    return new_graph, new_replacement_names


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
        warnings.warn(
            "The `GraphRewriter` class is deprecated and will be removed in the future. Use the `rewrite` function instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        new_graph, new_names = rewrite(
            original=graph,
            replacement=self._replacement,
            interface=self._interface,
            match=match,
            lhs=self._lhs,
            rhs=self._rhs,
        )
        return RewriteResult(
            new_graph=new_graph, pattern_to_original=match, replacement_to_new=new_names
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
