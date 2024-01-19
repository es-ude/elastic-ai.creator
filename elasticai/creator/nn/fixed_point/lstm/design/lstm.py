from functools import partial

from elasticai.creator.nn.fixed_point.linear.layer import Linear as FPLinear1d

from ._common_imports import (
    Design,
    InProjectTemplate,
    Path,
    Port,
    Signal,
    calculate_address_width,
    module_to_package,
    std_signals,
)


class LSTMNetworkDesign(Design):
    def __init__(
        self,
        lstm: Design,
        linear_layer: FPLinear1d,
        total_bits: int,
        frac_bits: int,
        hidden_size: int,
        input_size: int,
    ) -> None:
        super().__init__(name="lstm_network")
        self._linear_layer = linear_layer
        self._lstm = lstm
        self.template = InProjectTemplate(
            module_to_package(self.__module__),
            file_name="lstm_network.tpl.vhd",
            parameters=dict(
                name=self.name,
                lstm_cell_name=self._lstm.name,
                linear_name=self._linear_layer.name,
                data_width=str(total_bits),
                frac_width=str(frac_bits),
                hidden_size=str(hidden_size),
                input_size=str(input_size),
                linear_in_features=str(self._linear_layer.in_feature_num),
                linear_out_features=str(self._linear_layer.out_feature_num),
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
                Signal("x", width=lstm.port["x_data"].width),
                Signal("addr_in", width=lstm.port["h_out_addr"].width),
            ],
            outgoing=[
                std_signals.done(),
                Signal("d_out", width=lstm.port["h_out_data"].width),
            ],
        )
        self._subpath_name = "lstm_network"

    @property
    def port(self) -> Port:
        return self._port

    def save_to(self, destination: Path) -> None:
        self._lstm.save_to(destination)
        self._linear_layer.save_to(destination.create_subpath(self._linear_layer.name))
        destination.create_subpath(self._subpath_name).as_file(".vhd").write(
            self.template
        )
