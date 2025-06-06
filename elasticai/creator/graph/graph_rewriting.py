import copy
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Hashable, Protocol, TypeVar, runtime_checkable

from .graph import Graph
from .name_generation import NameRegistry


class DanglingEdgeError(Exception):
    def __init__(self, a, b):
        super().__init__(f"produced dangling edge {a} -> {b}")


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


TG = TypeVar("TG", bound=Hashable)
TR = TypeVar("TR", bound=Hashable)
TI = TypeVar("TI", bound=Hashable)

T = TypeVar("T", bound=Hashable)
TP = TypeVar("TP", bound=Hashable)


def get_rewriteable_matches(
    original: Graph[T], matches: Iterable[dict[TP, T]], interface_nodes: Iterable[TP]
) -> Iterator[dict[TP, T]]:
    """Yield all matches that do produce dangling edges and do not overlap with previous matches.

    The matches returned by this function are considerd safe to be rewritten in a single rewriting step in any order,
    without having to run an additional matching step and
    without producing dangling edges.

    :param original: The original graph.
    :param matches: The matches to check.
    :param interface_nodes: All nodes in the pattern that are belong to the interface. These nodes are considered to be preserved during rewriting. Thus, the edges connected to these nodes are not considered dangling.

    """
    interface_nodes = set(interface_nodes)
    matched_nodes_wo_interfaces: set[T] = set()

    for match in matches:
        if not produces_dangling_edge(original, match, interface_nodes):
            new_matched_nodes = set(match.values())
            new_matched_interface_nodes = {
                match[interface_node] for interface_node in interface_nodes
            }
            new_matched_nodes_wo_interfaces = (
                new_matched_nodes - new_matched_interface_nodes
            )

            if new_matched_nodes.isdisjoint(matched_nodes_wo_interfaces):
                matched_nodes_wo_interfaces.update(new_matched_nodes_wo_interfaces)
                yield match


def produces_dangling_edge(
    graph: Graph[T], match: dict[TP, T], interface_nodes: Iterable[TP]
) -> bool:
    """Check if there are dangling edges attached to non-interface nodes."""

    def is_dangling_edge(src: T, dst: T) -> bool:
        return src not in match.values() or dst not in match.values()

    def non_interface_nodes() -> Iterator[T]:
        for pn, gn in match.items():
            if pn not in interface_nodes:
                yield gn

    for gn in non_interface_nodes():
        for gp in graph.predecessors[gn]:
            if is_dangling_edge(gp, gn):
                return True
        for gs in graph.successors[gn]:
            if is_dangling_edge(gn, gs):
                return True
    return False


def rewrite(
    *,
    replacement: Graph[str],
    original: Graph[str],
    match: Mapping[str, str],
    lhs: Mapping[str, str],
    rhs: Mapping[str, str],
) -> tuple[Graph[str], dict[str, str]]:
    """Return new rewritten graph and a mapping of replacements to new nodes.

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

    :::

    :param replacement: The graph that will replace the matched pattern in the original graph.
    :param graph: The original graph where the pattern will be replaced.
    :param match: A dictionary mapping nodes from pattern to nodes in the original graph.
    :param lhs: Maps interface nodes to their corresponding nodes in the pattern (Left Hand Side).
    :param rhs: Maps interface nodes to their corresponding nodes in the replacement (Right Hand Side).
    :return: A tuple containing the new graph after the rewrite operation and a dictionary mapping replacement node names to new node names.
    :raises ValueError: If the `rhs` function is not injective (when different interface nodes map to the same replacement node).
    :raises DanglingEdgeError: if there is an edge between an unmatched node and a matched non-interface node.
    """
    rhs_inversed = {rhs[node]: node for node in rhs}
    replacement_nodes_in_interface = set(rhs_inversed.keys())
    if len(set(rhs.keys())) != len(replacement_nodes_in_interface):
        raise ValueError(
            "Ensure the `rhs` function is injective. The `rhs` function should map each interface node to a unique replacement node."
        )
    interface_nodes_in_pattern = set(lhs.values())
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
        if node not in interface_nodes_in_graph:
            for ns in original.successors[node]:
                if ns in nodes_to_be_removed:
                    raise DanglingEdgeError(node, ns)
            for np in original.predecessors[node]:
                if np in nodes_to_be_removed:
                    raise DanglingEdgeError(np, node)

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
