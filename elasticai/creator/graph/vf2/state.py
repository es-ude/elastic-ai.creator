from collections.abc import Iterator
from typing import Generic, TypeVar

from elasticai.creator.graph.graph import Graph

T = TypeVar("T")
TP = TypeVar("TP")


class State(Generic[T, TP]):
    def __init__(self, graph: Graph[T]) -> None:
        self.order: dict[T, int] = {n: i for i, n in enumerate(graph.nodes)}
        self.order_back: dict[int, T] = {v: k for k, v in self.order.items()}
        self.core: list[TP | None] = [None for _ in graph.nodes]
        self.in_nodes: list[int] = [0 for _ in graph.nodes]
        self.out_nodes: list[int] = [0 for _ in graph.nodes]
        self.graph: Graph[T] = graph
        self.current_depth = 0

    def remove_pair(self, a: T, b: TP) -> None:
        if self.current_depth == 1:
            self.in_nodes[self.order[a]] = 0
            self.out_nodes[self.order[a]] = 0
        self.core[self.order[a]] = None

    def contains_pair(self, a: T, b: TP) -> bool:
        return self.core[self.order[a]] == b

    def contains_node(self, node: T) -> bool:
        return self.core[self.order[node]] is not None

    def partial_nodes(self) -> set[T]:
        return set(self.iter_nodes())

    def partial_predecessors(self, n: T) -> set[T]:
        return set(self.graph.predecessors[n]).intersection(self.partial_nodes())

    def partial_successors(self, n: T) -> set[T]:
        return set(self.graph.successors[n]).intersection(self.partial_nodes())

    def unseen_nodes(self) -> set[T]:
        depth = self.current_depth
        unseen = set()
        for node, id in self.order.items():
            if not (
                self.in_nodes[id] == depth
                or self.out_nodes[id] == depth
                or self.core[id] is not None
            ):
                unseen.add(node)
        return unseen

    def in_node_successors(self, n: T) -> set[T]:
        nodes: set[T] = set()
        for n in self.graph.successors[n]:
            if self.in_nodes[self.order[n]] == self.current_depth:
                nodes.add(n)
        return nodes

    def in_node_predecessors(self, n: T) -> set[T]:
        nodes: set[T] = set()
        for n in self.graph.predecessors[n]:
            if self.in_nodes[self.order[n]] == self.current_depth:
                nodes.add(n)

        return nodes

    def out_node_successors(self, n: T) -> set[T]:
        nodes: set[T] = set()
        for n in self.graph.successors[n]:
            if self.out_nodes[self.order[n]] == self.current_depth:
                nodes.add(n)
        return nodes

    def out_node_predecessors(self, n: T) -> set[T]:
        nodes: set[T] = set()
        for n in self.graph.predecessors[n]:
            if self.out_nodes[self.order[n]] == self.current_depth:
                nodes.add(n)
        return nodes

    def unseen_successors(self, n: T) -> set[T]:
        return set(self.graph.successors[n]).intersection(self.unseen_nodes())

    def unseen_predecessors(self, n: T) -> set[T]:
        return set(self.graph.predecessors[n]).intersection(self.unseen_nodes())

    def iter_nodes(self) -> Iterator[T]:
        for n in self.order:
            if self.contains_node(n):
                yield n

    def is_complete(self) -> bool:
        return all(n is not None for n in self.core)

    def iter_matched_pairs(self) -> Iterator[tuple[T, TP]]:
        for node in self.graph.nodes:
            matched_node = self.core[self.order[node]]
            if matched_node is not None:
                yield node, matched_node

    def add_pair(self, a: T, b: TP) -> None:
        self.core[self.order[a]] = b
        self.in_nodes[self.order[a]] = self.current_depth
        self.out_nodes[self.order[a]] = self.current_depth

    def restore(self) -> None:
        """reset all changes introduced in the current depth.

        This is responsible for backtracking.
        """
        depth = self.current_depth
        for node_id, match in enumerate(self.core):
            if self.in_nodes[node_id] == depth:
                self.in_nodes[node_id] = 0
            if self.out_nodes[node_id] == depth:
                self.out_nodes[node_id] = 0
        self.current_depth -= 1

    def update_in_nodes(self) -> None:
        """Find out which nodes inside of current match are reachable from outside of current match."""
        depth = self.current_depth
        for n in self.order:
            if not self.contains_node(n):
                for nbr in self.graph.successors[n]:
                    if self.contains_node(nbr):
                        self.in_nodes[self.order[n]] = depth
                        break

    def update_out_nodes(self) -> None:
        """Find out which nodes outside of current match are reachable from the current match."""
        depth = self.current_depth
        for n in self.order:
            if not self.contains_node(n):
                for nbr in self.graph.predecessors[n]:
                    if self.contains_node(nbr):
                        self.out_nodes[self.order[n]] = depth
                        break

    def next_depth(self) -> None:
        self.current_depth += 1
        self.update_inout_nodes()

    def update_inout_nodes(self) -> None:
        self.update_in_nodes()
        self.update_out_nodes()

    def __iter_nodes(self, depth: int, in_out: list[int]) -> Iterator[T]:
        if depth == 0:
            return
        for node, node_id in self.order.items():
            added_at = in_out[node_id]
            if added_at >= depth:
                yield node

    def iter_in_nodes(self) -> Iterator[T]:
        depth = self.current_depth
        yield from self.__iter_nodes(depth, self.in_nodes)

    def iter_out_nodes(self) -> Iterator[T]:
        depth = self.current_depth
        yield from self.__iter_nodes(depth, self.out_nodes)
