from collections.abc import Sequence
from itertools import chain
from typing import Protocol

from elasticai.creator.vhdl.code import Code
from elasticai.creator.vhdl.connectable import Connectable
from elasticai.creator.vhdl.ports.signal_provider import BaseConnectableSignalProvider
from elasticai.creator.vhdl.signals import Signal
from elasticai.creator.vhdl.typing import Identifiable


class Port(Connectable["PortMap"], Protocol):
    def build_port_map(self, prefix: str) -> "PortMap":
        ...


class PortImpl(BaseConnectableSignalProvider, Port):
    def _id_of(self, s: Signal) -> str:
        return s.id()

    def __init__(self, in_signals, out_signals):
        super().__init__(receivers=out_signals, providers=in_signals)

    @property
    def _in_signals(self) -> Sequence[Signal]:
        return self._providers

    @property
    def _out_signals(self) -> Sequence[Signal]:
        return self._receivers

    def build_port_map(self, id: str) -> "PortMap":
        pm = PortMapImpl(
            id=id,
            in_signals=self._in_signals,
            out_signals=self._out_signals,
        )
        return pm


class PortMap(Connectable["Port"], Identifiable, Protocol):
    def signal_definitions(self) -> Code:
        ...

    def instantiation(self) -> Code:
        ...


class PortMapImpl(BaseConnectableSignalProvider, PortMap):
    def _id_of(self, s: Signal) -> str:
        return f"{self.id()}_{s.id()}"

    def __init__(
        self,
        id: str,
        in_signals: Sequence,
        out_signals: Sequence,
    ):
        super().__init__(
            receivers=in_signals,
            providers=out_signals,
        )
        self._id = id

    def id(self) -> str:
        return self._id

    def signal_definitions(self) -> Code:
        return [
            signal.definition(prefix=f"{self._id}_")
            for signal in chain(self._receivers, self._providers)
        ]

    def code(self) -> Code:
        result = [
            f"{self.id()} : entity work.{self.id()}(rtl)",
            "port map (",
        ]
        prefix = self.id()
        for port_signal in (x.id() for x in chain(self._receivers, self._providers)):
            result.append(f"{port_signal} => {prefix}_{port_signal},")
        result[-1] = result[-1].strip(",")
        result.append(");")
        return result
