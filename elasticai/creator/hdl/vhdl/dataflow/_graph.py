from typing import Generic, TypeVar

from .acceptor import Acceptor
from .data_flow_node import Node as Node

T = TypeVar("T")


class Graph(Generic[T]):
    def create_node(self, sources: list[T], sinks: list[Acceptor[T]]) -> Node[T]:
        return Node(sinks=sinks, sources=sources)
