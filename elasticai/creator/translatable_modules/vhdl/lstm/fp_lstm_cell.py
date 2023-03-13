from copy import copy
from functools import partial
from itertools import chain, repeat
from typing import Iterator, cast

import numpy as np

from elasticai.creator.hdl.code_generation.abstract_base_template import (
    TemplateConfig,
    TemplateExpander,
    module_to_package,
)
from elasticai.creator.hdl.code_generation.code_generation import (
    calculate_address_width,
)
from elasticai.creator.hdl.design_base.design import Design, Port
from elasticai.creator.hdl.design_base.signal import Signal
from elasticai.creator.hdl.translatable import Path
from elasticai.creator.hdl.vhdl.code_generation import to_vhdl_hex_string
from elasticai.creator.hdl.vhdl.code_generation.code_generation import (
    generate_hex_for_rom,
)
from elasticai.creator.hdl.vhdl.code_generation.template import Template
from elasticai.creator.nn._two_complement_fixed_point_config import (
    TwoComplementFixedPointConfig,
)


class FPLSTMCell(Design):
    def __init__(
        self,
        *,
        name: str,
        total_bits: int,
        frac_bits: int,
        w_ih: list[list[list[float]]],
        w_hh: list[list[list[float]]],
        b_ih: list[list[float]],
        b_hh: list[list[float]],
    ):
        super().__init__(
            name=name,
        )
        base_config = TemplateConfig(
            package=module_to_package(self.__module__), file_name="", parameters={}
        )
        self.input_size = len(w_ih[0])
        self.hidden_size = len(w_ih) // 4
        self.weights_ih = w_ih
        self.weights_hh = w_hh
        self.biases_ih = b_ih
        self.biases_hh = b_hh
        self._fp_config = TwoComplementFixedPointConfig(
            total_bits=total_bits, frac_bits=frac_bits
        )
        self._rom_base_config = copy(base_config)
        self._rom_base_config.file_name = "rom.tpl.vhd"

        self._config = copy(base_config)
        self._config.file_name = f"{self.name}.tpl.vhd"
        self._config.parameters = {
            k: str(v)
            for k, v in dict(
                name=self.name,
                libary="work",
                data_width=total_bits,
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                frac_width=frac_bits,
                x_h_addr_width=calculate_address_width(
                    self.input_size + self.hidden_size
                ),
                hidden_addr_width=calculate_address_width(self.hidden_size),
                w_addr_width=calculate_address_width(
                    (self.input_size + self.hidden_size) * self.hidden_size
                ),
            ).items()
        }

    @property
    def total_bits(self) -> int:
        return int(cast(str, self._config.parameters["data_width"]))

    @property
    def _hidden_addr_width(self) -> int:
        return int(cast(str, self._config.parameters["hidden_addr_width"]))

    @property
    def _weight_address_width(self) -> int:
        return int(cast(str, self._config.parameters["w_addr_width"]))

    @property
    def port(self) -> Port:
        ctrl_signal = partial(Signal, width=0)
        return Port(
            incoming=[
                Signal("x_data", self.total_bits),
                ctrl_signal("clock"),
                ctrl_signal("clk_hadamard"),
                ctrl_signal("reset"),
                ctrl_signal("zero_state"),
                ctrl_signal("h_out_en"),
                Signal("h_out_addr", self._hidden_addr_width),
            ],
            outgoing=[
                ctrl_signal("done"),
                Signal("h_out_data", self._hidden_addr_width),
            ],
        )

    def _build_weights(self):
        weights = np.concatenate((self.weights_ih, self.weights_hh), axis=1)
        weights = np.reshape(weights, (4, -1))
        w_i, w_f, w_g, w_o = weights.reshape(4, -1).tolist()

        bias = np.add(self.biases_ih, self.biases_hh)
        bias = np.reshape(bias, (4, -1))
        b_i, b_f, b_g, b_o = bias.tolist()

        def convert_floats_to_ints(floats: list[float]) -> list[int]:
            return list(map(self._fp_config.as_integer, floats))

        final_weights = tuple(map(convert_floats_to_ints, (w_i, w_f, w_g, w_o)))
        final_biases = tuple(map(convert_floats_to_ints, (b_i, b_f, b_g, b_o)))

        return final_weights, final_biases

    def save_to(self, destination: "Path"):
        weights, biases = self._build_weights()
        weights = zip(weights, ("wi", "wf", "wg", "wo"))
        biases = zip(biases, ("bi", "bf", "bg", "bo"))
        rom_template = TemplateExpander(self._rom_base_config)
        rom_values: list[str] | str = ["00"] * 2**self._weight_address_width
        rom_values = ",".join(map(generate_hex_for_rom, rom_values))
        for values, name in chain(weights, biases):
            self._rom_base_config.parameters.update(
                dict(
                    resource_option="auto",
                    values=values,
                    name=f"rom_{name}_{self.name}",
                    rom_addr_bitwidth=str(self._weight_address_width),
                    rom_data_bitwidth=str(self.total_bits),
                    rom_value=rom_values,
                )
            )
            rom_file = destination.create_subpath(f"{name}_rom").as_file(".vhd")
            rom_file.write_text(rom_template.lines())
        expander = TemplateExpander(self._config)
        destination.as_file(".vhd").write_text(expander.lines())


class _DualPortDoubleClockRom:
    def __init__(
        self,
        data_width: int,
        values: list[int],
        name: str,
        resource_option: str,
    ) -> None:
        self.name = name
        self.resource_option = resource_option
        self.data_width = data_width
        self.addr_width = calculate_address_width(len(values))
        padded_values = chain(values, repeat(0, 2**self.addr_width))

        def to_hex(number: int) -> str:
            return to_vhdl_hex_string(number=number, bit_width=self.data_width)

        self.hex_values = list(map(to_hex, padded_values))

    def lines(self) -> list[str]:
        template = Template(base_name="rom")
        template.update_parameters(
            name=self.name,
            rom_addr_bitwidth=str(self.addr_width),
            rom_data_bitwidth=str(self.data_width),
            rom_value=",".join(self.hex_values),
            rom_resource_option=f'"{self.resource_option}"',
        )

        return template.lines()
