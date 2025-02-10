from typing import Any

from elasticai.creator.ir import Edge, Graph
from elasticai.creator.ir import Node as _Node
from elasticai.creator.ir import node as _node


class Node(_Node):
    implementation: str


def new_node(
    name: str, type: str, implementation: str, attributes: dict[str, Any] | None = None
) -> Node:
    if attributes is None:
        attributes = {}
    return Node(
        _node(name, type, dict(implementation=implementation) | attributes).data
    )


def input_node(attributes: dict[str, Any] | None = None) -> Node:
    return new_node("input", "input", "input", attributes)


def output_node(attributes: dict[str, Any] | None = None) -> Node:
    return new_node("output", "output", "output", attributes)


class Implementation(Graph[Node, Edge]):
    name: str
    type: str
