from .ir_data import IrData
from .attribute import Attribute


class Node(IrData):
    name: str
    type: str


class Edge(IrData):
    src: str
    sink: str


def node(name: str, type: str, attributes: dict[str, Attribute] | None = None) -> Node:
    if attributes is None:
        attributes = dict()
    return Node(dict(name=name, type=type, **attributes))


def edge(src: str, sink: str, attributes: dict[str, Attribute] | None = None) -> Edge:
    if attributes is None:
        attributes = dict()
    return Edge(dict(src=src, sink=sink, **attributes))
