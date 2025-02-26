from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Any, TypeVar


@dataclass(eq=True, frozen=True)
class DataFlowNode:
    dsts: tuple[str, ...]
    sources: tuple[str, ...]
    name: str

    @classmethod
    def top(cls, name: str) -> "DataFlowNode":
        return cls(
            name=name,
            dsts=("done", "y", "x_address"),
            sources=("clock", "x", "y_address", "enable"),
        )

    @classmethod
    def buffered(cls, name: str) -> "DataFlowNode":
        return cls(
            name=name,
            dsts=("clock", "x", "y_address", "enable"),
            sources=("done", "y", "x_address"),
        )

    @classmethod
    def unbuffered(cls, name: str) -> "DataFlowNode":
        return cls(name=name, dsts=("clock", "x", "enable"), sources=("y",))


@dataclass(eq=True, frozen=True)
class WiringProtocol:
    """
    A wiring protocol specifies how we search for sources that need to be connected to dsts.
    We understand the neural network as a DAG. The direction "down" refers to the direction
    of the dataflow during inference. A "dst" refers to an "in" signal in vhdl and a source
    refers to an "out" signal. Each field of the class holds a dictionary specifying compatible
    source names for dsts. For "up_dsts" we search in ancestor nodes for satisfying sources
    and for "down_dsts" we search in successor nodes.
    """

    up_dsts: dict[str, tuple[str, ...]]
    down_dsts: dict[str, tuple[str, ...]]


BASIC_WIRING = WiringProtocol(
    up_dsts={
        "clock": ("clock",),
        "enable": ("enable", "done"),
        "x": ("x", "y"),
        "y": ("x", "y"),
        "done": ("enable", "done"),
    },
    down_dsts={
        "y_address": ("y_address", "x_address"),
        "x_address": ("y_address", "x_address"),
    },
)


class AutoWiringProtocolViolation(Exception):
    pass


T = TypeVar("T")


class AutoWirer:
    """
    Please note that this implementation performs the wiring solely based on names.
    Hence, it will not check whether signals are compatible based on their width
    or data type.
    It only supports a sequence of nodes as the input graph.
    Handling nodes with more than one child requires deciding how data paths are split
    and merged.
    """

    def __init__(self) -> None:
        self._reset()

    def _check_protocol_support_violations(self, graph, top):
        for node in chain(graph, (top,)):
            for dst in node.dsts:
                if dst not in self._all_protocol_dsts():
                    raise AutoWiringProtocolViolation(f"{dst}")
            for source in node.sources:
                if source not in self._all_protocol_sources():
                    raise AutoWiringProtocolViolation(f"{source}")

    def _all_protocol_sources(self) -> Iterator[str]:
        def flatten(items: Iterable[Iterable[str]]) -> Iterable[str]:
            return chain.from_iterable(items)

        down_sources = flatten(self._protocol.down_dsts.values())
        up_sources = flatten(self._protocol.up_dsts.values())
        return chain(down_sources, up_sources)

    def _all_protocol_dsts(self) -> Iterator[str]:
        return chain(self._protocol.up_dsts, self._protocol.down_dsts)

    def connections(self) -> dict[tuple[str, str], tuple[str, str]]:
        return self._connections

    def wire(self, top: DataFlowNode, graph: Iterable[DataFlowNode]):
        self._check_protocol_support_violations(graph=graph, top=top)
        self._reset()
        self._remember_sources_provided_by(top)
        self._remember_unsatisfied_down_dsts_from(top)
        g = chain(graph, (top,))
        for node in g:
            self._connect_all_down_dsts_for(node)
            self._connect_all_up_dsts_for(node)
            self._remember_sources_provided_by(node)
            self._remember_unsatisfied_down_dsts_from(node)

    def _remember_unsatisfied_down_dsts_from(self, node: "DataFlowNode"):
        for s in self._protocol_down_dsts(node):
            if (node.name, s) not in self._unsatisfied_down_dsts:
                self._unsatisfied_down_dsts.append((node.name, s))

    def _connect_all_down_dsts_for(self, node: "DataFlowNode"):
        unsatisfied_down_dsts = self._unsatisfied_down_dsts.copy()
        for node_name, dst_name in unsatisfied_down_dsts:
            for source in self._protocol_down_sources(dst_name, node):
                self._connect_down_dst(
                    dst=(node_name, dst_name), source=(node.name, source)
                )
                break

    def _connect_all_up_dsts_for(self, node: "DataFlowNode"):
        for dst in self._protocol_up_dsts(node):
            self._connect_up_dst(node.name, dst)

    def _update_available_sources(self, matched_dst: str, node_name: str, source: str):
        self._available_sources[matched_dst] = (node_name, source)

    def _remember_sources_provided_by(self, node: "DataFlowNode"):
        for dst in self._protocol.up_dsts:
            sources = self._protocol_up_sources(dst, node)
            for source in sources:
                self._update_available_sources(dst, node.name, source)

    def _reset(self) -> None:
        self._available_sources: dict[str, tuple[str, str]] = {}
        self._unsatisfied_down_dsts: list[tuple[str, str]] = list()
        self._connections: dict[tuple[str, str], tuple[str, str]] = {}
        self._protocol = BASIC_WIRING

    @staticmethod
    def _intersection_iterator(
        a: Iterable[T], b: Sequence[T] | dict[T, Any]
    ) -> Iterator[T]:
        for s in a:
            if s in b:
                yield s

    def _protocol_down_dsts(self, node) -> Iterator[str]:
        yield from self._intersection_iterator(node.dsts, self._protocol.down_dsts)

    def _protocol_up_dsts(self, node) -> Iterator[str]:
        yield from self._intersection_iterator(node.dsts, self._protocol.up_dsts)

    def _protocol_up_sources(self, dst: str, node: "DataFlowNode") -> Iterator[str]:
        yield from self._intersection_iterator(
            node.sources, self._protocol.up_dsts[dst]
        )

    def _protocol_down_sources(self, dst: str, node: "DataFlowNode") -> Iterator[str]:
        yield from self._intersection_iterator(
            node.sources, self._protocol.down_dsts[dst]
        )

    def _connect_up_dst(self, node_name: str, dst_name: str):
        self._connections[(node_name, dst_name)] = self._available_sources[dst_name]

    def _connect_down_dst(self, source: tuple[str, str], dst: tuple[str, str]):
        self._connections[dst] = source
        self._unsatisfied_down_dsts.remove(dst)
