from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from functools import singledispatchmethod
from typing import Generic, Protocol, TypeVar, overload

from .core import Edge
from .core import Node as Node

PatternNodeT = TypeVar("PatternNodeT", bound=Node)
GraphNodeT = TypeVar("GraphNodeT", bound=Node)
NodeT = TypeVar("NodeT", bound=Node)
EdgeT = TypeVar("EdgeT", bound=Edge)


class Graph(Protocol[NodeT, EdgeT]):
    def get_empty_copy(self) -> "Graph[NodeT, EdgeT]": ...

    def nodes(self) -> Mapping[str, NodeT]: ...

    def edges(self) -> Mapping[str, EdgeT]: ...

    def successors(self, node_name: str) -> Mapping[str, NodeT]: ...

    def predecessors(self, node_name: str) -> Mapping[str, NodeT]: ...


def find_subgraphs(
    graph: Graph[GraphNodeT, EdgeT],
    pattern: Graph[PatternNodeT, EdgeT],
    node_constraint: Callable[[GraphNodeT, PatternNodeT], bool],
) -> list[dict[str, str]]:
    """Find occurences of `pattern` in `graph` where nodes match according to `node_constraint`.

    :::{warning}
    This function is not optimized nor do we have benchmarks for performance. You might run into situations where runtime is too long for your use case.
    :::

    :::{important}
    The function will stop as soon as a match was found. That does not mean, that it will return at most one match.
    Instead matches that were found in the same iteration are returned together.
    :::

    :param: `graph`: The graph to search in.
    :param: `pattern`: Defines the pattern to search for. A node in the pattern will match if
       1. it has fewer or equal in- and out-edges as the corresponding `graph` node
       2. `node_constraint` holds for the corresponding node in `graph`
        pattern (Graph): _description_
       3. 1. and 2. hold for all predecessors and successors of the node
    :param: `node_constraint`: An additional constraint that must hold for a node in `graph` to match a node in `pattern`. If you just want to match the graph structure, independent of any node attributes, you can pass `lambda a, b: True` as the constraint.

    :return: A list of dictionaries, where each dictionary maps the names of the nodes in `pattern` to the names of the corresponding nodes in `graph`. The list contains one dictionary for each match found.
    """

    def stop_condition(matches: list[_Match]) -> bool:
        return any(m.is_complete() for m in matches) or len(matches) == 0

    return list(
        _SubgraphMatcher(
            graph, pattern, node_constraint, stop_condition=stop_condition
        ).find_matches()
    )


class _MatcherNodeMapping(Mapping[str, "_MatcherNode"]):
    def __init__(self, nodes: Mapping[str, Node], owner: Graph):
        self._owner = owner
        self._nodes = nodes

    def __getitem__(self, key: str) -> "_MatcherNode":
        return _MatcherNode(key, self._owner)

    def __iter__(self) -> Iterator[str]:
        return iter(self._nodes)

    def __contains__(self, key):
        return key in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)


class _MatcherNode(Generic[NodeT]):
    def __init__(self, name: str, owner: Graph[NodeT, EdgeT]):
        self.name = name
        self.owner = owner

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def in_degree(self) -> int:
        return len(self.predecessors)

    @property
    def out_degree(self) -> int:
        return len(self.successors)

    @property
    def unwrapped(self) -> NodeT:
        return self.owner.nodes[self.name]

    @property
    def successors(self) -> Mapping[str, "_MatcherNode"]:
        return _MatcherNodeMapping(self.owner.successors(self.name), self.owner)

    @property
    def predecessors(self) -> Mapping[str, "_MatcherNode"]:
        return _MatcherNodeMapping(self.owner.predecessors(self.name), self.owner)


class _Match(Generic[PatternNodeT, GraphNodeT]):
    def __init__(
        self,
        last_pattern_node: str,
        pattern: Graph[PatternNodeT, EdgeT],
        graph: Graph[GraphNodeT, EdgeT],
        map: dict[str, str] | None = None,
    ):
        if map is None:
            map = {}
        self.map = map
        self.pattern = pattern
        self.graph = graph
        self._last_pattern_node = last_pattern_node

    def last_pattern_node(self) -> _MatcherNode:
        return _MatcherNode(self._last_pattern_node, self.pattern)

    @property
    def pattern_successors(self) -> Mapping[str, _MatcherNode[PatternNodeT]]:
        return self.last_pattern_node().successors

    @property
    def graph_successors(self) -> Mapping[str, _MatcherNode[GraphNodeT]]:
        return self.last_graph_node().successors

    @property
    def pattern_predecessors(self) -> Mapping[str, _MatcherNode[PatternNodeT]]:
        return self.last_pattern_node().predecessors

    @property
    def graph_predecessors(self) -> Mapping[str, _MatcherNode[GraphNodeT]]:
        return self.last_graph_node().predecessors

    def last_graph_node(self) -> _MatcherNode:
        return _MatcherNode(self.map[self._last_pattern_node], self.graph)

    @singledispatchmethod
    def visited_pattern(self, pattern_node) -> bool:
        raise NotImplementedError(
            "Not implemented for type {}".format(type(pattern_node))
        )

    @visited_pattern.register
    def _(self, pattern_node: _MatcherNode) -> bool:
        return pattern_node.name in self.map

    @visited_pattern.register
    def _(self, pattern_node: str) -> bool:
        return self.visited_pattern(_MatcherNode(pattern_node, self.pattern))

    @singledispatchmethod
    def visited_graph(self, graph_node) -> bool:
        raise NotImplementedError(
            "Not implemented for type {}".format(type(graph_node))
        )

    @visited_graph.register
    def _(self, graph_node: _MatcherNode) -> bool:
        return graph_node.name in self.map.values()

    @visited_graph.register
    def _(self, graph_node: str) -> bool:
        return self.visited_graph(_MatcherNode(graph_node, self.graph))

    @overload
    def visited_neither(self, pattern_node: str, graph_node: str) -> bool: ...

    @overload
    def visited_neither(
        self,
        pattern_node: _MatcherNode[PatternNodeT],
        graph_node: _MatcherNode[GraphNodeT],
    ) -> bool: ...

    @singledispatchmethod  # type: ignore[operator, misc]
    def visited_neither(self, pattern_node, graph_node) -> bool:
        raise NotImplementedError(
            "Not implemented for types {} {}".format(
                type(pattern_node), type(graph_node)
            )
        )

    @visited_neither.register  # type: ignore[attr-defined]
    def _(
        self,
        pattern_node: _MatcherNode,
        graph_node: _MatcherNode,
    ) -> bool:
        return not (
            self.visited_pattern(pattern_node) or self.visited_graph(graph_node)
        )

    @visited_neither.register  # type: ignore[attr-defined]
    def _(self, pattern_node: str, graph_node: str) -> bool:
        return self.visited_neither(
            _MatcherNode(pattern_node, self.pattern),
            _MatcherNode(graph_node, self.graph),
        )

    @overload
    def visit(self, pattern_node: str, graph_node: str) -> _Match: ...

    @overload
    def visit(
        self,
        pattern_node: _MatcherNode[PatternNodeT],
        graph_node: _MatcherNode[GraphNodeT],
    ) -> _Match: ...

    @singledispatchmethod  # type: ignore[operator, misc]
    def visit(self, pattern_node, graph_node):
        raise NotImplementedError(
            "Not implemented for types {} {}".format(
                type(pattern_node), type(graph_node)
            )
        )

    @visit.register  # type: ignore[attr-defined]
    def _(
        self,
        pattern_node: _MatcherNode,
        graph_node: _MatcherNode,
    ):
        return _Match(
            pattern_node.name,
            self.pattern,
            self.graph,
            self.map | dict(((pattern_node.name, graph_node.name),)),
        )

    @visit.register  # type: ignore[attr-defined]
    def _(self, pattern_node: str, graph_node: str):
        return self.visit(
            _MatcherNode(pattern_node, self.pattern),
            _MatcherNode(graph_node, self.graph),
        )

    def is_complete(self) -> bool:
        return len(self.map) == len(self.pattern.nodes)


class _SubgraphMatcher(Generic[GraphNodeT, PatternNodeT]):
    def __init__(
        self,
        graph: Graph[GraphNodeT, EdgeT],
        pattern: Graph[PatternNodeT, EdgeT],
        node_constraint: Callable[[GraphNodeT, PatternNodeT], bool],
        stop_condition: Callable[[list[_Match]], bool],
    ):
        self._graph = graph
        self._pattern = pattern
        self._node_constraint = node_constraint
        self._stop_condition = stop_condition

    def find_matches(self) -> Iterator[dict[str, str]]:
        potential_matches = self._build_initial_potential_matches()
        while not self._stop_condition(potential_matches):
            potential_matches = self._build_next_potential_matches(potential_matches)
        matches = [pm.map for pm in potential_matches if pm.is_complete()]
        yield from matches

    def _build_initial_potential_matches(
        self,
    ) -> list[_Match[PatternNodeT, GraphNodeT]]:
        pn = self._pick_a_start_node_from_pattern()
        gns = []
        for name in self._find_all_start_nodes_from_graph(pn.name):
            gns.append(_MatcherNode(name, self._graph))

        potential_matches: list[_Match[PatternNodeT, GraphNodeT]] = []
        base_element = _Match("", self._pattern, self._graph)
        for gn in gns:
            potential_matches.append(base_element.visit(pn, gn))

        return potential_matches

    def _build_next_potential_matches(
        self, potential_matches: list[_Match[PatternNodeT, GraphNodeT]]
    ) -> list[_Match[PatternNodeT, GraphNodeT]]:
        next_potential_matches: list[_Match[PatternNodeT, GraphNodeT]] = []
        for potential_match in potential_matches:
            next_potential_matches.extend(
                self._find_follow_ups_for_match(potential_match)
            )
        return next_potential_matches

    def _find_follow_ups_for_match(
        self, potential_match: _Match[PatternNodeT, GraphNodeT]
    ) -> list[_Match[PatternNodeT, GraphNodeT]]:
        next_potential_matches: list[_Match[PatternNodeT, GraphNodeT]] = []

        def should_visit(pn, gn):
            return potential_match.visited_neither(pn, gn) and self._nodes_match(pn, gn)

        def visit(pn, gn):
            next_potential_matches.append(potential_match.visit(pn, gn))

        for ps in potential_match.pattern_successors:
            for gs in potential_match.graph_successors:
                if should_visit(ps, gs):
                    visit(ps, gs)

        for pp in potential_match.pattern_predecessors:
            for gp in potential_match.graph_predecessors:
                if should_visit(pp, gp):
                    visit(pp, gp)

        return next_potential_matches

    @overload
    def _nodes_match(self, pattern_node: str, graph_node: str) -> bool: ...

    @overload
    def _nodes_match(
        self,
        pattern_node: _MatcherNode[PatternNodeT],
        graph_node: _MatcherNode[GraphNodeT],
    ) -> bool: ...

    @singledispatchmethod  # type: ignore[operator, misc]
    def _nodes_match(self, pattern_node, graph_node) -> bool:
        raise NotImplementedError(
            "Not implemented for types {} {}".format(
                type(pattern_node), type(graph_node)
            )
        )

    @_nodes_match.register  # type: ignore[attr-defined]
    def _(
        self,
        pattern_node: _MatcherNode,
        graph_node: _MatcherNode,
    ) -> bool:
        return (
            self._check_constraint(pattern_node, graph_node)
            and graph_node.in_degree >= pattern_node.in_degree
            and graph_node.out_degree >= pattern_node.out_degree
        )

    def _check_constraint(
        self,
        pattern_node: _MatcherNode,
        graph_node: _MatcherNode,
    ) -> bool:
        return self._node_constraint(graph_node.unwrapped, pattern_node.unwrapped)

    # ignoring some errors as mypy can't handle overload+singledispatch
    @_nodes_match.register  # type: ignore[attr-defined, misc]
    def _(self, pattern_node: str, graph_node: str) -> bool:
        return self._nodes_match(
            _MatcherNode(pattern_node, self._pattern),
            _MatcherNode(graph_node, self._graph),
        )

    def _pick_a_start_node_from_pattern(self) -> _MatcherNode[PatternNodeT]:
        return _MatcherNode(next(iter(self._pattern.nodes.keys())), self._pattern)

    def _find_all_start_nodes_from_graph(self, start_node: str) -> set[str]:
        matches = set()
        _start_node = self._pattern.nodes[start_node]
        for node in self._graph.nodes.values():
            if self._node_constraint(node, _start_node):
                matches.add(node.name)
        return matches
