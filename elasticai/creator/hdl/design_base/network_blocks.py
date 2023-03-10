from abc import ABC

from elasticai.creator.hdl.code_generation.code_generation import (
    calculate_address_width,
)
from elasticai.creator.hdl.design_base import std_signals as _signals
from elasticai.creator.hdl.design_base.design import Design, Port


class NetworkBlock(Design, ABC):
    def __init__(self, name: str, x_width: int, y_width: int):
        super().__init__(name)
        self._port = Port(
            incoming=[
                _signals.enable(),
                _signals.clock(),
                _signals.x(x_width),
            ],
            outgoing=[_signals.y(y_width)],
        )


class BufferedNetworkBlock(Design, ABC):
    def __init__(
        self,
        name: str,
        x_width: int,
        y_width: int,
        x_count: int,
        y_count: int,
    ):
        super().__init__(name)
        in_signals = [
            _signals.enable(),
            _signals.clock(),
            _signals.x(x_width),
            _signals.y_address(calculate_address_width(y_count)),
        ]
        out_signals = [
            _signals.done(),
            _signals.y(y_width),
            _signals.x_address(calculate_address_width(x_count)),
        ]
        self._port = Port(incoming=in_signals, outgoing=out_signals)

    @property
    def port(self) -> Port:
        return self._port
