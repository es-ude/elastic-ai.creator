from itertools import chain, product
from typing import Generic, Iterator, Optional, TypeVar, cast

from elasticai.creator.vhdl.data_path_connection.typing import Sink

T = TypeVar("T")


class SourceNode(Generic[T]):
    def __init__(self, source: T, owner: "DataFlowNode[T]"):
        self._wrapped = source
        self._owner = owner
        self._sinks: set[SinkNode[T]] = set()

    def connect(self, child: "SinkNode[T]"):
        self._sinks.add(child)

    @property
    def owner(self) -> "DataFlowNode[T]":
        return self._owner

    @property
    def sinks(self) -> set["SinkNode[T]"]:
        return self._sinks

    @property
    def wrapped(self) -> T:
        return self._wrapped

    def __repr__(self):
        return f"SourceNode({self.wrapped})"


class SinkNode(Generic[T]):
    def __init__(self, sink: "Sink[T]", owner: "DataFlowNode[T]"):
        self._wrapped = sink
        self._owner = owner
        self._source: Optional[SourceNode[T]] = None

    def connect(self, source: SourceNode[T]):
        self._source = source

    def is_satisfied(self) -> bool:
        return self._source is not None

    @property
    def source(self) -> Optional[SourceNode[T]]:
        return self._source

    @property
    def owner(self) -> "DataFlowNode[T]":
        return self._owner

    @property
    def wrapped(self) -> "Sink[T]":
        return self._wrapped

    def accepts(self, source: SourceNode[T]) -> bool:
        return self.wrapped.accepts(source.wrapped)

    def __repr__(self):
        return f"SinkNode({self.wrapped})"


class DataFlowNode(Generic[T]):
    def __init__(self, sinks: set["Sink[T]"], sources: set[T]):
        self._sources: set[SourceNode[T]] = {
            SourceNode(source, self) for source in sources
        }
        self._sinks = {SinkNode(sink, self) for sink in sinks}

    @property
    def parents(self) -> set["DataFlowNode[T]"]:
        return {s.source.owner for s in self.sinks if s.source is not None}

    @property
    def children(self) -> set["DataFlowNode[T]"]:
        return {sink.owner for source in self.sources for sink in source.sinks}

    @property
    def sources(self) -> set[SourceNode[T]]:
        return {cast(SourceNode[T], source) for source in self._sources}

    @property
    def _unsatisfied_sinks(self) -> set[SinkNode[T]]:
        return {sink for sink in self._sinks if not sink.is_satisfied()}

    @property
    def _satisfied_sinks(self) -> set[SinkNode[T]]:
        return {sink for sink in self._sinks if sink.is_satisfied()}

    @property
    def sinks(self) -> set[SinkNode[T]]:
        return set(chain(self._unsatisfied_sinks, self._satisfied_sinks))

    @staticmethod
    def _matching_sink_source_pairs(
        sinks, sources
    ) -> Iterator[tuple[SinkNode[T], SourceNode[T]]]:
        return filter(lambda pair: pair[0].accepts(pair[1]), product(sinks, sources))

    def append(self, child: "DataFlowNode[T]") -> None:
        for sink, source in self._matching_sink_source_pairs(
            child._unsatisfied_sinks, self.sources
        ):
            sink.connect(source)
            source.connect(sink)

    def prepend(self, parent: "DataFlowNode[T]") -> None:
        for sink, source in self._matching_sink_source_pairs(
            self._unsatisfied_sinks, parent.sources
        ):
            sink.connect(source)
            source.connect(sink)

    def is_satisfied(self) -> bool:
        return len(self._unsatisfied_sinks) == 0

    def __repr__(self):
        return f"DataFlowNode(sinks={self.sinks}, sources={self.sources}"
