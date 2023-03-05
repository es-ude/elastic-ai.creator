from abc import ABC

from elasticai.creator.hdl.design_base import std_signals as _Signals
from elasticai.creator.hdl.design_base.design import Design, Port
from elasticai.creator.hdl.vhdl.code_generation.code_generation import (
    calculate_address_width,
)


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
    ):
        super().__init__(name)
        in_signals = [
            _Signals.enable(),
            _Signals.clock(),
            _Signals.x(x_width),
            _Signals.y_address(calculate_address_width(x_width)),
        ]
        out_signals = [
            _Signals.done(),
            _Signals.y(y_width),
            _Signals.x_address(calculate_address_width(y_width)),
        ]
        self._port = Port(incoming=in_signals, outgoing=out_signals)
