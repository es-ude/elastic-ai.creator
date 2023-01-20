from elasticai.creator.vhdl.port import Port
from elasticai.creator.vhdl.signals import (
    LogicInSignal,
    LogicInVectorSignal,
    LogicOutSignal,
    LogicOutVectorSignal,
)
from elasticai.creator.vhdl.vhdl_design import BaseVHDLDesign


class BaseHWBlock(BaseVHDLDesign):
    def __init__(self, x_width: int, y_width: int):
        super().__init__("", template_names=tuple())
        self._x_width = x_width
        self._y_width = y_width

    def get_port(self) -> Port:
        in_signals: list[LogicInSignal | LogicInVectorSignal] = [
            LogicInSignal(name, default_value="'0'") for name in ("enable", "clock")
        ]
        in_signals.append(LogicInVectorSignal("x", width=self._x_width))
        return Port(
            in_signals=in_signals,
            out_signals=[LogicOutVectorSignal("y", width=self._y_width)],
        )


class BufferedBaseHWBlock(BaseHWBlock):
    def __init__(
        self,
        x_width: int,
        y_width: int,
        x_address_width: int,
        y_address_width: int,
    ):
        super().__init__(y_width=y_width, x_width=x_width)
        self._x_address_width = x_address_width
        self._y_address_width = y_address_width

    def get_port(self) -> Port:
        in_signals = [
            LogicInSignal("enable", default_value="'0'"),
            LogicInSignal("clock", default_value="'0'"),
            LogicInVectorSignal("y_address", self._y_address_width),
            LogicInVectorSignal("x", self._x_width),
        ]
        out_signals = [
            LogicOutSignal("done", default_value="'0'"),
            LogicOutVectorSignal("y", self._y_width),
            LogicOutVectorSignal("x_address", self._x_address_width),
        ]
        return Port(in_signals=in_signals, out_signals=out_signals)
