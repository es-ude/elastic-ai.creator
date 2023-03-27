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
from elasticai.creator.nn.vhdl.fp_linear_1d.design import FPLinear1d


class LSTMNetworkDesign(Design):
    def __init__(
        self,
        lstm: Design,
        linear_layers: list[FPLinear1d],
        total_bits: int,
        frac_bits: int,
        hidden_size: int,
        input_size: int,
    ) -> None:
        super().__init__(name="lstm_network")
        self._linear_layers = linear_layers
        self._lstm = lstm
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

        ctrl_signal = partial(Signal, width=0)
        self._port = Port(
            incoming=[
                std_signals.clock(),
                std_signals.enable(),
                ctrl_signal("x_we"),
                Signal("x_in", width=lstm.port["x_data"].width),
                Signal("addr_in", width=lstm.port["h_out_addr"].width),
            ],
            outgoing=[
                std_signals.done(),
                Signal("d_out", width=lstm.port["h_out_data"].width),
            ],
        )

    @property
    def port(self) -> Port:
        return self._port

    def save_to(self, destination: Path) -> None:
        self._lstm.save_to(destination.create_subpath("lstm_cell"))
        for index, layer in enumerate(self._linear_layers):
            layer.save_to(destination.create_subpath(f"fp_linear_1d_{index}"))
        expander = TemplateExpander(self.config)
        destination.create_subpath("lstm_network").as_file(".vhd").write_text(
            expander.lines()
        )
