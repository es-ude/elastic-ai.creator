from typing import TypeVar

from elasticai.creator.graph._types import NodeConstraintFn

from .vf2 import find_all_matches, match

TP = TypeVar("TP")
T = TypeVar("T")


class SubGraphMatchError(Exception):
    def __init__(self):
        super().__init__("No matching subgraph found")


def find_subgraph(
    pattern, graph, node_constraint: NodeConstraintFn[TP, T]
) -> dict[TP, T]:
    return match(pattern, graph, node_constraint)


def find_all_subgraphs(
    pattern, graph, node_constraint: NodeConstraintFn[TP, T]
) -> list[dict[TP, T]]:
    return find_all_matches(pattern, graph, node_constraint)
