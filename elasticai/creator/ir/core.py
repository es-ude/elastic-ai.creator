from .attribute import Attribute
from .ir_data import IrData
from .required_field import ReadOnlyField


def _read_only_str() -> ReadOnlyField[str, str]:
    return ReadOnlyField(lambda x: x)


class Node(IrData):
    """
    NOTE: name and type are read only when accessed through their descriptor, i.e.
    >>> n = Node(dict())
    >>> n.name = "x" #  Error! read only
    >>> x = n.name  #  Error! no key `name` in n.data
    >>> n.data['name'] = "x"
    >>> n.name #  Ok!
    ... 'x'
    """

    name: ReadOnlyField[str, str] = _read_only_str()
    type: ReadOnlyField[str, str] = _read_only_str()


class Edge(IrData):
    """
    NOTE: src, sink are read only when accessed through their descriptors.
    """

    src: ReadOnlyField[str, str] = _read_only_str()
    sink: ReadOnlyField[str, str] = _read_only_str()


def node(name: str, type: str, attributes: dict[str, Attribute] | None = None) -> Node:
    if attributes is None:
        attributes = dict()
    return Node(dict(name=name, type=type, **attributes))


def edge(src: str, sink: str, attributes: dict[str, Attribute] | None = None) -> Edge:
    if attributes is None:
        attributes = dict()
    return Edge(dict(src=src, sink=sink, **attributes))
