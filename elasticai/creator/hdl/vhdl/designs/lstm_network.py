from functools import partial

from elasticai.creator.hdl.design_base.design import Design, Port
from elasticai.creator.hdl.design_base.signal import Signal


class LSTMNetwork(Design):
    def save_to(self, destination: "Path"):
        pass

    def __init__(self, x_width: int, y_address_width: int, y_width: int):
        super().__init__(name="lstm_network")
        signal = partial(Signal, accepted_names=[])
        ctrl_signal = partial(Signal, accepted_names=[], width=0)
        self._port = Port(
            incoming=[
                ctrl_signal("clock"),
                ctrl_signal("enable"),
                ctrl_signal("x_we"),
                signal("x_in", x_width),
                signal("addr_in", y_address_width),
            ],
            outgoing=[
                ctrl_signal("done"),
                signal("d_out", y_width),
            ],
        )

    @property
    def port(self) -> Port:
        return self._port


class LSTMNetworkSkeleton(Design):
    def __init__(self, data_width, data_address_width):
        super().__init__("lstm_network")
        self._port = self._create_port(data_width, data_address_width)

    def _create_port(self, data_width: int, data_address_width: int) -> Port:
        _signal = partial(Signal, accepted_names=[])
        incoming_control_signals = [
            _signal(name=name, width=0)
            for name in ("clock", "clk_hadamard", "reset", "rd", "wr")
        ]
        incoming_data_signals = [
            _signal(name="data_in", width=data_width),
            _signal(name="address_in", width=data_address_width),
        ]
        outgoing_data_signals = [
            _signal(name="data_out", width=data_width),
            _signal(name="debug", width=8),
            _signal(name="led_ctrl", width=4),
        ]
        outgoing_control_signals = [
            _signal(name="busy", width=0),
            _signal(name="wake_up", width=0),
        ]
        return Port(
            incoming=incoming_data_signals + incoming_control_signals,
            outgoing=outgoing_control_signals + outgoing_data_signals,
        )

    @property
    def port(self) -> Port:
        return self._port

    def save_to(self, destination: "Path"):
        pass
