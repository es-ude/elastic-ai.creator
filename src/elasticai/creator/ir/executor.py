from collections.abc import Callable, Iterator
from typing import Protocol

import elasticai.creator.function_dispatch as FD
from elasticai.creator.graph import bfs_iter_down

from .datagraph import DataGraph as _DataGraph
from .datagraph import Edge, Node


class DataGraph[N: Node, E: Edge](_DataGraph[N, E], Protocol):
    @property
    def type(self) -> str: ...


type Handler[G: DataGraph[Node, Edge], R] = Callable[[G, R], R]


class ExecutionOrderGraphReducer[N: Node, R]:
    """Call handlers on each node N in execution order, reducing a result R."""

    def __init__(self):
        self._input_nodes: tuple[N, ...] = tuple()

    def __call__(self, g: DataGraph[N, Edge], start_val: R) -> R:
        """Call handlers on each node N in execution order, reducing a result R."""
        self._collect_input_nodes(g)
        val = start_val
        for node in self._yield_nodes_in_dependency_order(g):
            val = self._call_handler(node, val)
        return val  # ty: ignore

    def _collect_input_nodes(self, g: DataGraph[N, Edge]) -> None:
        nodes = []
        for n in g.nodes.values():
            if n.type == "input":
                nodes.append(n)
        self._input_nodes = tuple(nodes)

    def _yield_nodes_in_dependency_order(self, g: DataGraph[N, Edge]) -> Iterator[N]:
        def succ(n: str) -> Iterator[str]:
            yield from g.successors[n]

        def pred(n: str) -> Iterator[str]:
            yield from g.predecessors[n]

        yield from self._input_nodes

        for name in bfs_iter_down(succ, pred, set(n.name for n in self._input_nodes)):
            yield g.nodes[name]

    @FD.dispatch_method()
    def _call_handler(self, fn: Callable[[N, R], R], node: N, previous_result: R) -> R:
        return fn(node, previous_result)

    @_call_handler.key_from_args
    def _get_key_from_arg(self, node: N, previous_result: R) -> str:
        return node.type

    @_call_handler.default_override
    def override(self, key: str | None, fn: Callable[[N, R], R]) -> Callable[[N, R], R]:
        return fn

    @_call_handler.default_register
    def register(self, key: str | None, fn: Callable[[N, R], R]) -> Callable[[N, R], R]:
        return fn
