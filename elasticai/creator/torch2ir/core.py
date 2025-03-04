from typing import Any

from elasticai.creator.ir import Attribute, Edge, GraphProtocol
from elasticai.creator.ir import Implementation as _Implementation
from elasticai.creator.ir import Node as _Node


class Node(_Node):
    implementation: str


def new_node(
    name: str, type: str, implementation: str, attributes: dict[str, Any] | None = None
) -> Node:
    if attributes is None:
        attributes = {}
    return Node(name, dict(type=type, implementation=implementation) | attributes)


def input_node(attributes: dict[str, Any] | None = None) -> Node:
    return new_node("input", "input", "input", attributes)


def output_node(attributes: dict[str, Any] | None = None) -> Node:
    return new_node("output", "output", "output", attributes)


class Implementation(_Implementation[Node, Edge]):
    name: str
    type: str

    def __init__(
        self,
        *,
        graph: GraphProtocol,
        name: str | None = None,
        type: str | None = None,
        data: dict[str, Attribute] | None = None,
    ):
        super().__init__(data=data, graph=graph, node_fn=Node, edge_fn=Edge)
        if name is not None:
            self.name = name
        if type is not None:
            self.type = type
