from typing import Iterable

from elasticai.creator.vhdl.dataflow.node_connector import ConnectableNode


class FakeNodeFactory:
    def __init__(self):
        self.required_signals = 0
        self.next_id = 0

    def create(self, children: list["FakeConnectableNode"]) -> "FakeConnectableNode":
        i = FakeConnectableNode(
            children=children,
            required_signals=self.required_signals,
            id=self.next_id,
        )
        self.next_id += 1
        return i


class FakeConnectableNode(ConnectableNode):
    def is_missing_inputs(self) -> bool:
        return len(self.connections) < self.required_signals

    @property
    def parents(self: "FakeConnectableNode") -> Iterable["FakeConnectableNode"]:
        return self._parents

    @property
    def children(self) -> Iterable["FakeConnectableNode"]:
        return self._children

    def __init__(
        self, children: Iterable["FakeConnectableNode"], required_signals: int, id: int
    ):
        self._children = list(children)
        self._parents: list["FakeConnectableNode"] = []
        self._id = id
        for child in children:
            child._parents.append(self)
        self.connection_complete = False
        self.required_signals = required_signals
        self.connections: list["FakeConnectableNode"] = []

    def id(self) -> str:
        return f"{self._id}"

    @property
    def name(self) -> str:
        return self.id()

    def __repr__(self):
        return f"Node(id={self.id})"

    def connect(self, other: "FakeConnectableNode") -> None:
        self.connections.append(other)
