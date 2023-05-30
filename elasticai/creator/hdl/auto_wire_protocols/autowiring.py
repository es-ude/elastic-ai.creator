from dataclasses import dataclass
from itertools import chain


@dataclass(eq=True, frozen=True)
class DataFlowNode:
    sinks: tuple[str, ...]
    sources: tuple[str, ...]
    name: str

    @classmethod
    def top(cls, name: str) -> "DataFlowNode":
        return cls(
            name=name,
            sinks=("done", "y", "x_address"),
            sources=("clock", "x", "y_address", "enable"),
        )

    @classmethod
    def buffered(cls, name: str) -> "DataFlowNode":
        return cls(
            name=name,
            sinks=("clock", "x", "y_address", "enable"),
            sources=("done", "y", "x_address"),
        )

    @classmethod
    def unbuffered(cls, name: str) -> "DataFlowNode":
        return cls(name=name, sinks=("clock", "x", "enable"), sources=("y",))


@dataclass(eq=True, frozen=True)
class WiringProtocol:
    """
    A wiring protocol specifies how we search for sources that need to be connected to sinks.
    We understand the neural network as a DAG. The direction "down" refers to the direction
    of the dataflow during inference. A "sink" refers to an "in" signal in vhdl and a source
    refers to an "out" signal. Each field of the class holds a dictionary specifying compatible
    source names for sinks. For "up_sinks" we search in ancestor nodes for satisfying sources
    and for "down_sinks" we search in successor nodes.
    """

    up_sinks: dict[str, tuple[str, ...]]
    down_sinks: dict[str, tuple[str, ...]]


BASIC_WIRING = WiringProtocol(
    up_sinks={
        "clock": ("clock",),
        "enable": ("enable", "done"),
        "x": ("x", "y"),
        "y": ("x", "y"),
        "done": ("enable", "done"),
    },
    down_sinks={
        "y_address": ("y_address", "x_address"),
        "x_address": ("y_address", "x_address"),
    },
)


class AutoWirer:
    def __init__(self) -> None:
        self._reset()

    def connections(self) -> dict[tuple[str, str], tuple[str, str]]:
        return self._connections

    def wire(self, top: DataFlowNode, graph: tuple[DataFlowNode, ...]):
        self._reset()
        self._remember_sources_provided_by(top)
        self._remember_unsatisfied_down_sinks_from(top)
        g = chain(graph, (top,))
        for node in g:
            self._connect_all_up_sinks_for(node)
            self._connect_all_down_sinks_for(node)
            self._remember_sources_provided_by(node)
            self._remember_unsatisfied_down_sinks_from(node)

    def _remember_unsatisfied_down_sinks_from(self, node: "DataFlowNode"):
        self._unsatisfied_down_sinks.update(
            {(node.name, s) for s in self._protocol_down_sinks(node)}
        )

    def _connect_all_down_sinks_for(self, node: "DataFlowNode"):
        unsatisfied_down_sinks = self._unsatisfied_down_sinks.copy()
        for node_name, sink_name in unsatisfied_down_sinks:
            for source in self._protocol_down_sources(sink_name, node):
                self._connect_down_sink(
                    sink=(node_name, sink_name), source=(node.name, source)
                )
                break

    def _connect_all_up_sinks_for(self, node: "DataFlowNode"):
        for sink in self._protocol_up_sinks(node):
            self._connect_up_sink(node.name, sink)

    def _update_available_sources(self, matched_sink: str, node_name: str, source: str):
        self._available_sources[matched_sink] = (node_name, source)

    def _remember_sources_provided_by(self, node: "DataFlowNode"):
        for sink in self._protocol.up_sinks:
            sources = self._protocol_up_sources(sink, node)
            for source in sources:
                self._update_available_sources(sink, node.name, source)

    def _reset(self) -> None:
        self._available_sources: dict[str, tuple[str, str]] = {}
        self._unsatisfied_down_sinks: set[tuple[str, str]] = set()
        self._connections: dict[tuple[str, str], tuple[str, str]] = {}
        self._protocol = BASIC_WIRING

    def _protocol_down_sinks(self, node) -> set[str]:
        return set(node.sinks).intersection(self._protocol.down_sinks)

    def _protocol_up_sinks(self, node) -> set[str]:
        return set(node.sinks).intersection(self._protocol.up_sinks)

    def _protocol_up_sources(self, sink: str, node: "DataFlowNode") -> set[str]:
        return set(self._protocol.up_sinks[sink]).intersection(node.sources)

    def _protocol_down_sources(self, sink: str, node: "DataFlowNode") -> set[str]:
        return set(node.sources).intersection(self._protocol.down_sinks[sink])

    def _connect_up_sink(self, node_name: str, sink_name: str):
        self._connections[(node_name, sink_name)] = self._available_sources[sink_name]

    def _connect_down_sink(self, source: tuple[str, str], sink: tuple[str, str]):
        self._connections[sink] = source
        self._unsatisfied_down_sinks.remove(sink)
