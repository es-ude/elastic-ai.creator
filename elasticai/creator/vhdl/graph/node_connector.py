from typing import Iterable, Protocol

from elasticai.creator.vhdl.connectable import Connectable
from elasticai.creator.vhdl.graph.graph import ancestors_breadth_first
from elasticai.creator.vhdl.graph.typing import Node


class ConnectableNode(Node, Connectable, Protocol):
    ...


class BreadthFirstNodeConnector:
    def __init__(self, nodes: Iterable[ConnectableNode]):
        self._nodes = nodes

    def connect(self) -> None:
        for node in self._nodes:
            for ancestor in ancestors_breadth_first(node):
                if not node.is_missing_inputs():
                    break
                else:
                    node.connect(ancestor)
