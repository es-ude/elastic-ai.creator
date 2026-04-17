from collections.abc import Collection, Iterable, Iterator, Mapping
from typing import Hashable, Protocol, Self, TypeVar

from .name_generation import NameRegistry
from .vf2.graph import Graph


class DanglingEdgeError(Exception):
    def __init__(self, a, b):
        super().__init__(f"produced dangling edge {a} -> {b}")


TG = TypeVar("TG", bound=Hashable)
TR = TypeVar("TR", bound=Hashable)
TI = TypeVar("TI", bound=Hashable)

T = TypeVar("T", bound=Hashable)
TP = TypeVar("TP", bound=Hashable)


def get_rewriteable_matches(
    original: Graph[T],
    matches: Iterable[dict[TP, T]],
    interface_nodes: Iterable[TP],
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


class _Graph(Graph[str], Protocol):
    def new(self) -> Self: ...
    @property
    def nodes(self) -> Collection[str]: ...
    def add_node(self, node: str) -> None: ...
    def iter_edges(self) -> Iterable[tuple[str, str]]: ...
    def add_edge(self, src: str, dst: str) -> None: ...
