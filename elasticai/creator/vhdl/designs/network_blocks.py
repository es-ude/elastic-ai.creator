from elasticai.creator.vhdl.language.ports import Port, PortImpl
from elasticai.creator.vhdl.language.signals import Signal
from elasticai.creator.vhdl.language.signals import SignalBuilder as _SignalBuilder
from elasticai.creator.vhdl.language.vhdl_template import VHDLTemplate

from .vhdl_design import BaseVHDLDesign


class _Signals:
    builder_logic_signals = _SignalBuilder().default("'0'")
    builder_vector_signals = _SignalBuilder()

    @classmethod
    def enable(cls) -> Signal:
        return (
            cls.builder_logic_signals.id("enable")
            .width(0)
            .accepted_names(["enable", "done"])
            .build()
        )

    @classmethod
    def clock(cls) -> Signal:
        return cls.builder_logic_signals.id("clock").accepted_names(["clock"]).build()

    @classmethod
    def done(cls) -> Signal:
        return (
            cls.builder_logic_signals.id("done")
            .accepted_names(["enable", "done"])
            .build()
        )

    @classmethod
    def x(cls, width: int) -> Signal:
        return (
            cls.builder_vector_signals.id("x")
            .accepted_names(["x", "y"])
            .width(width)
            .build()
        )

    @classmethod
    def y(cls, width: int) -> Signal:
        return (
            cls.builder_vector_signals.id("y")
            .accepted_names(["x", "y"])
            .width(width)
            .build()
        )

    @classmethod
    def x_address(cls, width: int) -> Signal:
        return (
            cls.builder_vector_signals.id("x_address")
            .accepted_names(["x_address", "y_address"])
            .width(width)
            .build()
        )

    @classmethod
    def y_address(cls, width: int) -> Signal:
        return (
            cls.builder_vector_signals.id("y_address")
            .accepted_names(["x_address", "y_address"])
            .width(width)
            .build()
        )


class NetworkBlock(BaseVHDLDesign):
    def __init__(self, name: str, template_name: str, x_width: int, y_width: int):
        self._main_file = VHDLTemplate(name=name, template_name=template_name)
        super().__init__(name=name, files=(self._main_file,))
        self._x_width = x_width
        self._y_width = y_width

    def get_port(self) -> Port:
        return PortImpl(
            in_signals=[
                _Signals.enable(),
                _Signals.clock(),
                _Signals.x(self._x_width),
            ],
            out_signals=[_Signals.y(self._y_width)],
        )


class BufferedNetworkBlock(NetworkBlock):
    def __init__(
        self,
        name: str,
        template_name: str,
        x_width: int,
        y_width: int,
        x_address_width: int,
        y_address_width: int,
    ):
        super().__init__(
            name=name, template_name=template_name, x_width=x_width, y_width=y_width
        )
        self._x_address_width = x_address_width
        self._y_address_width = y_address_width

    def get_port(self) -> Port:
        in_signals: list[Signal] = [
            _Signals.enable(),
            _Signals.clock(),
            _Signals.x(self._x_width),
            _Signals.y_address(self._y_address_width),
        ]
        out_signals: list[Signal] = [
            _Signals.done(),
            _Signals.y(self._y_width),
            _Signals.x_address(self._x_address_width),
        ]
        return PortImpl(in_signals=in_signals, out_signals=out_signals)
