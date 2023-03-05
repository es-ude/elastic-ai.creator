from itertools import chain, product
from typing import Generic, Iterator, Optional, TypeVar, cast

from elasticai.creator.hdl.design_base.acceptor import Acceptor

T = TypeVar("T")


class Source(Generic[T]):
    def __init__(self, source: T, owner: "Node[T]"):
        self._wrapped = source
        self._owner = owner
        self._sinks: list[Sink[T]] = list()

    def _connect(self, child: "Sink[T]"):
        self._sinks.append(child)

    @property
    def owner(self) -> "Node[T]":
        return self._owner

    @property
    def sinks(self) -> list["Sink[T]"]:
        return self._sinks

    @property
    def data(self) -> T:
        return self._wrapped

    def __repr__(self):
        return f"SourceNode({self.data})"


class Sink(Generic[T]):
    def __init__(self, sink: "Acceptor[T]", owner: "Node[T]"):
        self._wrapped = sink
        self._owner = owner
        self._source: Optional[Source[T]] = None

    def connect(self, source: Source[T]):
        self._source = source

    def is_satisfied(self) -> bool:
        return self._source is not None

    @property
    def source(self) -> Optional[Source[T]]:
        return self._source

    @property
    def owner(self) -> "Node[T]":
        return self._owner

    @property
    def data(self) -> "Acceptor[T]":
        return self._wrapped

    def accepts(self, source: Source[T]) -> bool:
        return self.data.accepts(source.data)

    def __repr__(self):
        return f"SinkNode({self.data})"


class Node(Generic[T]):
    def __init__(self, sinks: list["Acceptor[T]"], sources: list[T]):
        self._sources: set[Source[T]] = {Source(source, self) for source in sources}
        self._sinks = {Sink(sink, self) for sink in sinks}

    @property
    def parents(self) -> set["Node[T]"]:
        return {s.source.owner for s in self.sinks if s.source is not None}

    @property
    def children(self) -> set["Node[T]"]:
        return {sink.owner for source in self.sources for sink in source.sinks}

    @property
    def sources(self) -> set[Source[T]]:
        return {cast(Source[T], source) for source in self._sources}

    @property
    def _unsatisfied_sinks(self) -> set[Sink[T]]:
        return {sink for sink in self._sinks if not sink.is_satisfied()}

    @property
    def _satisfied_sinks(self) -> set[Sink[T]]:
        return {sink for sink in self._sinks if sink.is_satisfied()}

    @property
    def sinks(self) -> set[Sink[T]]:
        return set(chain(self._unsatisfied_sinks, self._satisfied_sinks))

    @staticmethod
    def _matching_sink_source_pairs(
        sinks, sources
    ) -> Iterator[tuple[Sink[T], Source[T]]]:
        return filter(lambda pair: pair[0].accepts(pair[1]), product(sinks, sources))

    def append(self, child: "Node[T]") -> None:
        for sink, source in self._matching_sink_source_pairs(
            child._unsatisfied_sinks, self.sources
        ):
            sink.connect(source)
            source._connect(sink)

    def prepend(self, parent: "Node[T]") -> None:
        for sink, source in self._matching_sink_source_pairs(
            self._unsatisfied_sinks, parent.sources
        ):
            sink.connect(source)
            source._connect(sink)

    def is_satisfied(self) -> bool:
        return len(self._unsatisfied_sinks) == 0

    def __repr__(self):
        return f"DataFlowNode(sinks={self.sinks}, sources={self.sources}"
