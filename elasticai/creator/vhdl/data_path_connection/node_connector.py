from abc import abstractmethod
from typing import Iterable, Protocol, TypeVar

from elasticai.creator.vhdl.data_path_connection.node_iteration import (
    ancestors_breadth_first,
)
from elasticai.creator.vhdl.graph import Node

T_Connectable_contra = TypeVar(
    "T_Connectable_contra", bound="Connectable", contravariant=True
)
T_contra = TypeVar("T_contra", contravariant=True)


class Connectable(Protocol[T_contra]):
    @abstractmethod
    def connect(self: T_Connectable_contra, other: T_contra):
        ...

    @abstractmethod
    def is_missing_inputs(self) -> bool:
        ...


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
