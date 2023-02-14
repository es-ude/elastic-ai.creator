from itertools import chain
from typing import Iterator, Sequence

from elasticai.creator.vhdl.code import Code
from elasticai.creator.vhdl.signals import Signal

from .typing import PortMap


class PortMapImpl(PortMap):
    def _id_of(self, s: Signal) -> str:
        return f"{self.id()}_{s.id()}"

    def __init__(
        self,
        id: str,
        in_signals: Sequence,
        out_signals: Sequence,
    ):
        self._in_signals = in_signals
        self._out_signals = out_signals
        self._id = id

    def id(self) -> str:
        return self._id

    @property
    def _signals(self) -> Iterator[Signal]:
        return chain(self._in_signals, self._out_signals)

    def signal_definitions(self) -> Code:
        return [signal.definition(prefix=f"{self._id}_") for signal in self._signals]

    def instantiation(self) -> Code:
        result = [
            f"{self.id()} : entity work.{self.id()}(rtl)",
            "port map (",
        ]
        prefix = self.id()
        for port_signal in (x.id() for x in self._signals):
            result.append(f"{port_signal} => {prefix}_{port_signal},")
        result[-1] = result[-1].strip(",")
        result.append(");")
        return result
