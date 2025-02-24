from typing import Protocol, TypeVar

from elasticai.creator import ir

PNodeCon = TypeVar("PNodeCon", bound=ir.Node, contravariant=True)
GNodeCon = TypeVar("GNodeCon", bound=ir.Node, contravariant=True)


class NodeConstraint(Protocol[PNodeCon, GNodeCon]):
    def __call__(self, *, pattern_node: PNodeCon, graph_node: GNodeCon) -> bool: ...
