from abc import abstractmethod
from itertools import chain
from typing import Generic, Iterable, Optional, Protocol, TypeVar

T_contra = TypeVar("T_contra", contravariant=True)
T = TypeVar("T")
T_Sink = TypeVar("T_Sink", bound="Sink")


class Sink(Protocol[T_contra]):
    @abstractmethod
    def accepts(self, source: T_contra) -> bool:
        ...


class SourceNode(Protocol[T_Sink, T]):
    @property
    @abstractmethod
    def owner(self) -> "DataFlowNode":
        ...

    @property
    @abstractmethod
    def sinks(self) -> Iterable["SinkNode[T_Sink, T]"]:
        ...

    @property
    @abstractmethod
    def wrapped(self) -> T:
        ...


class SinkNode(Protocol[T_Sink, T]):
    @property
    @abstractmethod
    def source(self) -> Optional[SourceNode[T_Sink, T]]:
        ...

    @property
    @abstractmethod
    def owner(self) -> "DataFlowNode[T_Sink, T]":
        ...

    @property
    @abstractmethod
    def wrapped(self) -> T_Sink:
        ...


class _SourceNode(SourceNode[T_Sink, T]):
    def __init__(self, source: T, owner: "DataFlowNode[T_Sink, T]"):
        self._wrapped = source
        self._owner = owner
        self._sinks: set[SinkNode[T_Sink, T]] = set()

    def connect(self, child: SinkNode[T_Sink, T]):
        self._sinks.add(child)

    @property
    def owner(self) -> "DataFlowNode[T_Sink, T]":
        return self._owner

    @property
    def sinks(self) -> Iterable[SinkNode[T_Sink, T]]:
        return self._sinks

    @property
    def wrapped(self) -> T:
        return self._wrapped

    def __repr__(self):
        return f"SourceNode({self.wrapped})"


class _SinkNode(SinkNode[T_Sink, T]):
    def __init__(self, sink: T_Sink, owner: "DataFlowNode[T_Sink, T]"):
        self._wrapped = sink
        self._owner = owner
        self._source: Optional[_SourceNode[T_Sink, T]] = None

    def connect(self, source: _SourceNode[T_Sink, T]):
        self._source = source

    def is_satisfied(self) -> bool:
        return self._source is not None

    @property
    def source(self) -> Optional[SourceNode[T_Sink, T]]:
        return self._source

    @property
    def owner(self) -> "DataFlowNode[T_Sink, T]":
        return self._owner

    @property
    def wrapped(self) -> T_Sink:
        return self._wrapped

    def accepts(self, source: _SourceNode[T_Sink, T]) -> bool:
        return self.wrapped.accepts(source.wrapped)

    def __repr__(self):
        return f"SinkNode({self.wrapped})"


class DataFlowNode(Generic[T_Sink, T]):
    def __init__(self, sinks: set[T_Sink], sources: set[T]):
        self._sources: set[_SourceNode[T_Sink, T]] = {
            _SourceNode(source, self) for source in sources
        }
        self._sinks = {_SinkNode(sink, self) for sink in sinks}

    @property
    def parents(self) -> set["DataFlowNode[T_Sink, T]"]:
        return {s.source.owner for s in self.sinks if s.source is not None}

    @property
    def children(self) -> set["DataFlowNode[T_Sink, T]"]:
        return {sink.owner for source in self.sources for sink in source.sinks}

    @property
    def sources(self) -> set[SourceNode[T_Sink, T]]:
        return self._sources

    @property
    def _unsatisfied_sinks(self) -> set[_SinkNode[T_Sink, T]]:
        return {sink for sink in self._sinks if not sink.is_satisfied()}

    @property
    def _satisfied_sinks(self) -> set[_SinkNode[T_Sink, T]]:
        return {sink for sink in self._sinks if sink.is_satisfied()}

    @property
    def sinks(self) -> set[SinkNode[T_Sink, T]]:
        return set(chain(self._unsatisfied_sinks, self._satisfied_sinks))

    def append(self, child: "DataFlowNode[T_Sink, T]") -> None:
        for sink in child._unsatisfied_sinks:
            for source in self._sources:
                if sink.accepts(source):
                    source.connect(sink)
                    sink.connect(source)
                    break

    def is_satisfied(self) -> bool:
        return len(self._unsatisfied_sinks) == 0

    def __repr__(self):
        return f"DataFlowNode(sinks={self.sinks}, sources={self.sources}"
