from typing import TypeVar

from elasticai.creator.graph._types import NodeConstraintFn

from .vf2 import match as _match

TP = TypeVar("TP")
T = TypeVar("T")


def find_subgraph(
    pattern, graph, node_constraint: NodeConstraintFn[TP, T]
) -> dict[TP, T]:
    return _match(pattern, graph, node_constraint)


def find_subgraphs(
    pattern, graph, node_constraint: NodeConstraintFn[TP, T]
) -> list[dict[TP, T]]:
    return [_match(pattern, graph, node_constraint)]
