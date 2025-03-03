from elasticai.creator.ir.base import Attribute, IrData, ReadOnlyField


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

    type: ReadOnlyField[str, str] = _read_only_str()
    __slots__ = ("name",)

    def __init__(self, name: str, data: dict[str, Attribute]) -> None:
        super().__init__(data)
        self.name = name


class Edge(IrData):
    """
    NOTE: src, dst are read only when accessed through their descriptors.
    """

    __slots__ = ("src", "dst")

    def __init__(self, src: str, dst: str, data: dict[str, Attribute]) -> None:
        super().__init__(data)
        self.src = src
        self.dst = dst


def node(name: str, type: str, attributes: dict[str, Attribute] | None = None) -> Node:
    if attributes is None:
        attributes = dict()
    return Node(name, dict(type=type, **attributes))


def edge(src: str, dst: str, attributes: dict[str, Attribute] | None = None) -> Edge:
    if attributes is None:
        attributes = dict()
    return Edge(src=src, dst=dst, data=attributes)
