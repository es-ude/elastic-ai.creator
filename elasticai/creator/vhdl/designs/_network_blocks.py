from elasticai.creator.vhdl.hardware_description_language.ports import Port
from elasticai.creator.vhdl.hardware_description_language.signals import Signal

from .design import Design


class _Signals:
    @classmethod
    def _logic_signal(cls, id: str, accepted_names: list[str]) -> Signal:
        return Signal(id=id, accepted_names=[id] + accepted_names, width=0)

    @classmethod
    def _vector_signal(cls, id: str, accepted_names: list[str], width: int) -> Signal:
        return Signal(id=id, accepted_names=[id] + accepted_names, width=width)

    @classmethod
    def enable(cls) -> Signal:
        id = "enable"
        return cls._logic_signal(id, ["done"])

    @classmethod
    def clock(cls) -> Signal:
        return cls._logic_signal("clock", [])

    @classmethod
    def done(cls) -> Signal:
        return cls._logic_signal("done", ["enable"])

    @classmethod
    def x(cls, width: int) -> Signal:
        return cls._vector_signal("x", ["y"], width)

    @classmethod
    def y(cls, width: int) -> Signal:
        return cls._vector_signal("y", ["x"], width)

    @classmethod
    def x_address(cls, width: int) -> Signal:
        return cls._vector_signal("x_address", ["y_address"], width)

    @classmethod
    def y_address(cls, width: int) -> Signal:
        return cls._vector_signal("y_address", ["x_address"], width)


class NetworkBlock(Design):
    def __init__(self, name: str, x_width: int, y_width: int):
        super().__init__(name=name)
        self._x_width = x_width
        self._y_width = y_width

    def port(self) -> Port:
        return Port(
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
        x_width: int,
        y_width: int,
        x_address_width: int,
        y_address_width: int,
    ):
        super().__init__(name=name, x_width=x_width, y_width=y_width)
        self._x_address_width = x_address_width
        self._y_address_width = y_address_width

    def port(self) -> Port:
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
        return Port(in_signals=in_signals, out_signals=out_signals)
