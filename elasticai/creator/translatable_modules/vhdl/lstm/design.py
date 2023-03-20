from functools import partial

from elasticai.creator.hdl.code_generation.abstract_base_template import (
    TemplateConfig,
    TemplateExpander,
    module_to_package,
)
from elasticai.creator.hdl.code_generation.code_generation import (
    calculate_address_width,
)
from elasticai.creator.hdl.design_base import std_signals
from elasticai.creator.hdl.design_base.design import Design, Port
from elasticai.creator.hdl.design_base.signal import Signal
from elasticai.creator.hdl.translatable import Path
from elasticai.creator.translatable_modules.vhdl.fp_linear_1d.design import FPLinear1d


class LSTMNetworkDesign(Design):
    def save_to(self, destination: "Path"):
        self._lstm.save_to(destination.create_subpath("lstm_cell"))
        for index, layer in enumerate(self._linear_layers):
            layer.save_to(destination.create_subpath(f"fp_linear_1d_{index}"))
        expander = TemplateExpander(self.config)
        destination.create_subpath("lstm_network").as_file(".vhd").write_text(
            expander.lines()
        )

    def __init__(
        self,
        lstm: Design,
        linear_layers: list[FPLinear1d],
        total_bits: int,
        frac_bits: int,
        hidden_size: int,
        input_size: int,
    ):
        super().__init__(name="lstm_network")
        self._linear_layers = linear_layers
        signal = partial(Signal)
        self._lstm = lstm
        ctrl_signal = partial(Signal, width=0)
        self.config = TemplateConfig(
            module_to_package(self.__module__),
            file_name="lstm_network.tpl.vhd",
            parameters=dict(
                data_width=str(total_bits),
                frac_width=str(frac_bits),
                hidden_size=str(hidden_size),
                input_size=str(input_size),
                linear_in_features=str(self._linear_layers[0].in_feature_num),
                linear_out_features=str(self._linear_layers[0].out_feature_num),
                hidden_addr_width=(
                    f"{calculate_address_width(hidden_size + input_size)}"
                ),
                x_h_addr_width=f"{calculate_address_width(hidden_size + input_size)}",
                w_addr_width=(
                    f"{calculate_address_width((hidden_size + input_size) * hidden_size)}"
                ),
                in_addr_width="4",
            ),
        )
        self._port = Port(
            incoming=[
                std_signals.clock(),
                std_signals.enable(),
                ctrl_signal("x_we"),
                signal("x_in", lstm.port["x_data"].width),
                signal("addr_in", lstm.port["h_out_addr"].width),
            ],
            outgoing=[
                ctrl_signal("done"),
                signal("d_out", lstm.port["h_out_data"].width),
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
        _signal = Signal
        ctr_signal = partial(Signal, width=0)
        incoming_control_signals = [
            ctr_signal(name=name)
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
            ctr_signal(name="busy"),
            ctr_signal(name="wake_up"),
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
