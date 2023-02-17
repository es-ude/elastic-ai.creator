from elasticai.creator.vhdl.language.signals import Signal

from ._port_map_impl import PortMapImpl
from ._typing import Port, PortMap


class PortImpl(Port):
    @staticmethod
    def _id_of(s: Signal) -> str:
        return s.id()

    def __init__(self, in_signals, out_signals):
        self._in_signals = in_signals
        self._out_signals = out_signals
        super().__init__(receivers=out_signals, providers=in_signals)

    def build_port_map(self, id: str) -> "PortMap":
        pm = PortMapImpl(
            id=id,
            in_signals=self._in_signals,
            out_signals=self._out_signals,
        )
        return pm
