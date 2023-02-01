from abc import ABC, abstractmethod
from typing import Generic, Iterator, Sequence, TypeVar

from elasticai.creator.vhdl.code import Code
from elasticai.creator.vhdl.signals import Signal

T = TypeVar("T")


class Connection(Generic[T]):
    def __init__(self, source: T, destination: T):
        self.source = source
        self.destination = destination


class BaseConnectableSignalProvider(ABC):
    def connect(self, other: "BaseConnectableSignalProvider"):
        self._connect_signal_providers(other)
        if isinstance(other, BaseConnectableSignalProvider):
            other._connect_signal_providers(self)

    def is_missing_inputs(self) -> bool:
        return len(self._unsatisfied_receivers) > 0

    def __init__(
        self,
        receivers: Sequence[Signal],
        providers: Sequence[Signal],
    ):
        self._unsatisfied_receivers = set(receivers)
        self._connections: list[Connection] = list()
        self._receivers = set(receivers)
        self._providers = set(providers)

    def connections(self) -> Code:
        def _iter() -> Iterator[str]:
            for connection in self._connections:
                yield f"{connection.destination} <= {connection.source};"

        return _iter()

    @abstractmethod
    def _id_of(self, s: Signal) -> str:
        ...

    def _connect_signal_providers(self, right: "BaseConnectableSignalProvider"):
        if isinstance(right, self.__class__):
            connections = []
            satisfied_receivers = set()
            for receiver in self._unsatisfied_receivers:
                for provider in right._providers:
                    if receiver.accepts(provider):
                        satisfied_receivers.add(receiver)
                        connections.append(
                            Connection(
                                self._id_of(receiver),
                                right._id_of(provider),
                            )
                        )
                        break
            self._connections = connections
            self._unsatisfied_receivers.difference_update(satisfied_receivers)
