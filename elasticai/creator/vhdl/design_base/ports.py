from collections.abc import Iterator
from itertools import chain

from elasticai.creator.vhdl.design_base.signal import Signal


class Port:
    def __init__(self, incoming: list[Signal], outgoing: list[Signal]):
        self._incoming = set(incoming)
        self._outgoing = set(outgoing)

    @property
    def incoming(self) -> list[Signal]:
        return list(self._incoming)

    @property
    def outgoing(self) -> list[Signal]:
        return list(self._outgoing)

    @property
    def signals(self) -> list[Signal]:
        return self.outgoing + self.incoming

    @property
    def signal_names(self) -> list[str]:
        return [s.name for s in self.signals]

    def __getitem__(self, item: str) -> Signal:
        for signal in self.signals:
            if signal.name == item:
                return signal
        raise AttributeError(f"Port has no signal {item}")

    def __contains__(self, item: str | Signal) -> bool:
        if isinstance(item, str):
            return item in self.signal_names
        else:
            return item in self.signals

    def __iter__(self) -> Iterator[Signal]:
        yield from chain(self._incoming, self._outgoing)
