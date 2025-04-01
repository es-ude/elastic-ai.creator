from collections.abc import Callable, Iterator
from functools import singledispatchmethod
from typing import Generic, Protocol, TypeVar, overload

from elasticai.creator.graph.graph import Graph

T = TypeVar("T", int, str)
TPattern = TypeVar("TPattern", int, str)

_Tcon = TypeVar("_Tcon", contravariant=True)
_TPatternCon = TypeVar("_TPatternCon", contravariant=True)


class NodeConstraint(Protocol[_TPatternCon, _Tcon]):
    def __call__(self, *, pattern_node: _TPatternCon, graph_node: _Tcon) -> bool: ...


def find_subgraphs(
    *,
    pattern: Graph[TPattern],
    graph: Graph[T],
    node_constraint: NodeConstraint[TPattern, T],
) -> list[dict[TPattern, T]]:
    """Find occurences of `pattern` in `graph` where nodes match according to `node_constraint`.

    Use the `node_constraint` function to define how nodes in `graph` and `pattern` should match.
    Even though the function only takes the node identifiers as arguments, you can still compare more complex objects, e.g.,
    as follows:

    ```python
    graph_nodes_data = {
        "a": {"type": "A", "channels": 1},
        "b": {"type": "B", "channels": 2},
    }
    pattern_nodes_data = {
        "0": {"type": "A", "channels": 1},
        "1": {"type": "B", "channels": 2},
    }

    def node_constraint(pattern_node: str, graph_node: str) -> bool:
        return graph_nodes_data[graph_node]["type"] == pattern_nodes_data[pattern_node]["type"]

    subgraphs = find_subgraphs(graph, pattern, node_constraint)
    ```

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

    def stop_condition(matches: list[_Match[TPattern, T]]) -> bool:
        return any(m.is_complete() for m in matches) or len(matches) == 0

    return _SubgraphMatcher(
        pattern, graph, node_constraint, stop_condition=stop_condition
    ).find_matches()


class _MatcherNode(Generic[T]):
    def __init__(self, name: T, owner: Graph[T]):
        self.name: T = name
        self.owner: Graph[T] = owner

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def in_degree(self) -> int:
        return len(self.owner.predecessors[self.name])

    @property
    def out_degree(self) -> int:
        return len(self.owner.successors[self.name])

    @property
    def successors(self) -> Iterator["_MatcherNode[T]"]:
        for node in self.owner.successors[self.name]:
            yield _MatcherNode(node, self.owner)

    @property
    def predecessors(self) -> Iterator["_MatcherNode[T]"]:
        for node in self.owner.predecessors[self.name]:
            yield _MatcherNode(node, self.owner)


class _Match(Generic[TPattern, T]):
    def __init__(
        self,
        last_pattern_node: TPattern | None,
        pattern: Graph[TPattern],
        graph: Graph[T],
        map: dict[TPattern, T] | None = None,
    ):
        if map is None:
            map = {}
        self.map: dict[TPattern, T] = map
        self.pattern: Graph[TPattern] = pattern
        self.graph: Graph[T] = graph
        self._last_pattern_node: TPattern | None = last_pattern_node

    def _get_last_pattern_node(self) -> _MatcherNode[TPattern]:
        assert self._last_pattern_node is not None
        return _MatcherNode(self._last_pattern_node, self.pattern)

    @property
    def pattern_successors(self) -> Iterator[_MatcherNode[TPattern]]:
        return self._get_last_pattern_node().successors

    @property
    def graph_successors(self) -> Iterator[_MatcherNode[T]]:
        return self._last_graph_node().successors

    @property
    def pattern_predecessors(self) -> Iterator[_MatcherNode[TPattern]]:
        return self._get_last_pattern_node().predecessors

    @property
    def graph_predecessors(self) -> Iterator[_MatcherNode[T]]:
        return self._last_graph_node().predecessors

    def _last_graph_node(self) -> _MatcherNode[T]:
        assert self._last_pattern_node is not None
        return _MatcherNode(self.map[self._last_pattern_node], self.graph)

    @singledispatchmethod
    def visited_pattern(self, pattern_node) -> bool:
        return self.visited_pattern(_MatcherNode(pattern_node, self.pattern))

    @visited_pattern.register
    def _(self, pattern_node: _MatcherNode) -> bool:
        return pattern_node.name in self.map

    @singledispatchmethod
    def visited_graph(self, graph_node) -> bool:
        return self.visited_graph(_MatcherNode(graph_node, self.graph))

    @visited_graph.register
    def _(self, graph_node: _MatcherNode) -> bool:
        return graph_node.name in self.map.values()

    @overload
    def visited_neither(self, pattern_node: T, graph_node: T) -> bool: ...

    @overload
    def visited_neither(
        self,
        pattern_node: _MatcherNode[TPattern],
        graph_node: _MatcherNode[T],
    ) -> bool: ...

    @singledispatchmethod  # type: ignore[operator, misc]
    def visited_neither(self, pattern_node, graph_node) -> bool:
        return self.visited_neither(
            _MatcherNode(pattern_node, self.pattern),
            _MatcherNode(graph_node, self.graph),
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

    @overload
    def visit(self, pattern_node: TPattern, graph_node: T) -> "_Match[T, TPattern]": ...

    @overload
    def visit(
        self,
        pattern_node: _MatcherNode[TPattern],
        graph_node: _MatcherNode[T],
    ) -> "_Match[TPattern, T]": ...

    @singledispatchmethod  # type: ignore[operator, misc]
    def visit(self, pattern_node, graph_node):
        return self.visit(
            _MatcherNode(pattern_node, self.pattern),
            _MatcherNode(graph_node, self.graph),
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

    def is_complete(self) -> bool:
        return len(self.map) == len(self.pattern.nodes)


class _SubgraphMatcher(Generic[TPattern, T]):
    def __init__(
        self,
        pattern: Graph[TPattern],
        graph: Graph[T],
        node_constraint: NodeConstraint[TPattern, T],
        stop_condition: Callable[[list[_Match[TPattern, T]]], bool],
    ):
        self._graph: Graph[T] = graph
        self._pattern: Graph[TPattern] = pattern
        self._node_constraint: NodeConstraint[TPattern, T] = node_constraint
        self._stop_condition: Callable[[list[_Match[TPattern, T]]], bool] = (
            stop_condition
        )

    def find_matches(self) -> list[dict[TPattern, T]]:
        potential_matches = self._build_initial_potential_matches()
        while not self._stop_condition(potential_matches):
            potential_matches = self._build_next_potential_matches(potential_matches)
        matches = [pm.map for pm in potential_matches if pm.is_complete()]
        return matches

    def _build_initial_potential_matches(
        self,
    ) -> list[_Match[TPattern, T]]:
        pn = self._pick_a_start_node_from_pattern()
        gns = []
        for name in self._find_all_start_nodes_from_graph(pn.name):
            gns.append(_MatcherNode(name, self._graph))

        potential_matches: list[_Match[TPattern, T]] = []
        base_element = _Match(None, self._pattern, self._graph)
        for gn in gns:
            potential_matches.append(base_element.visit(pn, gn))

        return potential_matches

    def _build_next_potential_matches(
        self, potential_matches: list[_Match[TPattern, T]]
    ) -> list[_Match[TPattern, T]]:
        next_potential_matches: list[_Match[TPattern, T]] = []
        for potential_match in potential_matches:
            next_potential_matches.extend(
                self._find_follow_ups_for_match(potential_match)
            )
        return next_potential_matches

    def _find_follow_ups_for_match(
        self, potential_match: _Match[TPattern, T]
    ) -> list[_Match[TPattern, T]]:
        next_potential_matches: list[_Match[TPattern, T]] = []

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
    def _nodes_match(self, pattern_node: TPattern, graph_node: T) -> bool: ...

    @overload
    def _nodes_match(
        self,
        pattern_node: _MatcherNode[TPattern],
        graph_node: _MatcherNode[T],
    ) -> bool: ...

    @singledispatchmethod  # type: ignore[operator, misc]
    def _nodes_match(
        self, pattern_node: _MatcherNode[TPattern], graph_node: _MatcherNode[T]
    ) -> bool:
        return self._node_constraint(graph_node.name, pattern_node.name)

    # ignoring some errors as mypy can't handle overload+singledispatch
    @singledispatchmethod  # type: ignore[operator, misc]
    def _nodes_match(self, pattern_node, graph_node) -> bool:
        return self._nodes_match(
            _MatcherNode(pattern_node, self._pattern),
            _MatcherNode(graph_node, self._graph),
        )

    @_nodes_match.register  # type: ignore[attr-defined, misc]
    def _nodes_match(
        self, pattern_node: _MatcherNode, graph_node: _MatcherNode
    ) -> bool:
        return self._node_constraint(
            graph_node=graph_node.name, pattern_node=pattern_node.name
        )

    def _pick_a_start_node_from_pattern(self) -> _MatcherNode[TPattern]:
        return _MatcherNode(next(iter(self._pattern.nodes)), self._pattern)

    def _find_all_start_nodes_from_graph(self, start_node: TPattern) -> set[T]:
        matches = set()
        for node in self._graph.nodes:
            if self._node_constraint(pattern_node=start_node, graph_node=node):
                matches.add(node)
        return matches
