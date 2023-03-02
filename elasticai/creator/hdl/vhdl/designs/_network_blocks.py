from abc import ABC

from elasticai.creator.hdl.vhdl.designs import std_signals as _Signals

from ..signal import Signal
from .design import Design, Port


class NetworkBlock(Design, ABC):
    def __init__(self, name: str, x_width: int, y_width: int):
        super().__init__(name)
        self._port = Port(
            incoming=[
                _Signals.enable(),
                _Signals.clock(),
                _Signals.x(x_width),
            ],
            outgoing=[_Signals.y(y_width)],
        )

    @property
    def port(self) -> Port:
        return self._port


class BufferedNetworkBlock(Design, ABC):
    def __init__(
        self,
        name: str,
        x_width: int,
        y_width: int,
        x_address_width: int,
        y_address_width: int,
    ):
        super().__init__(name)
        in_signals = [
            _Signals.enable(),
            _Signals.clock(),
            _Signals.x(x_width),
            _Signals.y_address(y_address_width),
        ]
        out_signals = [
            _Signals.done(),
            _Signals.y(y_width),
            _Signals.x_address(x_address_width),
        ]
        self._port = Port(incoming=in_signals, outgoing=out_signals)
