import math

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port
from elasticai.creator.vhdl.shared_designs.rom import (
    Rom,
)

from .testbench import Conv1dDesignProtocol


def generate_parameters_from_port(port: Port) -> dict[str, str]:
    params = {}
    for signal in port:
        if signal.width > 0:
            params[f"{signal.name}_width"] = str(signal.width)
    return params


class Conv1dDesign(Design, Conv1dDesignProtocol):
    def __init__(
        self,
        name: str,
        total_bits: int,
        frac_bits: int,
        in_channels: int,
        out_channels: int,
        signal_length: int,
        kernel_size: int,
        weights: list[list[list[int]]],
        bias: list[int],
    ) -> None:
        super().__init__(name=name)
        self._total_bits = total_bits
        self._frac_bits = frac_bits
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._input_signal_length = signal_length
        self._kernel_size = kernel_size
        self._weights = weights
        self._bias = bias
        self.output_signal_length = math.floor(
            self.input_signal_length - self.kernel_size + 1
        )
        self._port = create_port(
            x_width=self._total_bits,
            y_width=self._total_bits,
            x_count=self.input_signal_length * self._in_channels,
            y_count=self.output_signal_length * self._out_channels,
        )

    @property
    def input_signal_length(self) -> int:
        return self._input_signal_length

    @property
    def kernel_size(self) -> int:
        return self._kernel_size

    @property
    def port(self) -> Port:
        return self._port

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @staticmethod
    def _flatten_params(params: list[list[list[int]]]) -> list[int]:
        result = []
        for list_of_lists in params:
            for list_of_int in list_of_lists:
                result.extend(list_of_int)
        return result

    def save_to(self, destination: Path) -> None:
        print(self.name)
        rom_name = dict(weights=f"{self.name}_w_rom", bias=f"{self.name}_b_rom")
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="conv1d.tpl.vhd",
            parameters=dict(
                frac_width=str(self._frac_bits),
                in_channels=str(self._in_channels),
                out_channels=str(self._out_channels),
                kernel_size=str(self.kernel_size),
                vector_width=str(self.input_signal_length),
                name=self.name,
            )
            | generate_parameters_from_port(self._port),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        core_component = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="conv1d_function.tpl.vhd",
            parameters={
                "rom_name_weights": rom_name["weights"],
                "rom_name_bias": rom_name["bias"],
            },
        )
        destination.create_subpath(f"{self.name}_fxp_MAC_RoundToZero").as_file(
            ".vhd"
        ).write(core_component)

        mac = InProjectTemplate(
            package="elasticai.creator.nn.fixed_point.mac",
            file_name="fxp_mac.tpl.vhd",
            parameters={},
        )
        destination.create_subpath("fxp_mac").as_file(".vhd").write(mac)

        weights_rom = Rom(
            name=rom_name["weights"],
            data_width=self._total_bits,
            values_as_integers=self._flatten_params(self._weights),
        )
        weights_rom.save_to(destination)

        bias_rom = Rom(
            name=rom_name["bias"],
            data_width=self._total_bits,
            values_as_integers=self._bias,
        )
        bias_rom.save_to(destination)
