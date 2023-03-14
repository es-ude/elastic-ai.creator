from copy import copy
from functools import partial
from itertools import repeat
from typing import Any, cast

import numpy as np

from elasticai.creator.hdl.code_generation.abstract_base_template import (
    TemplateConfig,
    TemplateExpander,
    module_to_package,
)
from elasticai.creator.hdl.code_generation.code_generation import (
    calculate_address_width,
    to_hex,
)
from elasticai.creator.hdl.design_base.design import Design, Port
from elasticai.creator.hdl.design_base.signal import Signal
from elasticai.creator.hdl.translatable import Path
from elasticai.creator.hdl.vhdl.code_generation.code_generation import (
    generate_hex_for_rom,
)
from elasticai.creator.hdl.vhdl.designs import HardSigmoid
from elasticai.creator.nn._two_complement_fixed_point_config import (
    TwoComplementFixedPointConfig,
)
from elasticai.creator.translatable_modules.vhdl.lstm.fp_hard_tanh import FPHardTanh


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

        rom_template = TemplateExpander(self._rom_base_config)
        write_files = partial(
            self._write_files, destination=destination, rom_template=rom_template
        )
        write_files(
            names=("wi", "wf", "wg", "wo"),
            parameters=weights,
            address_width=self._weight_address_width,
        )
        write_files(
            names=("bi", "bf", "bg", "bo"),
            parameters=biases,
            address_width=self._hidden_addr_width,
        )
        self._save_dual_port_double_clock_ram(destination)
        self._save_hardtanh(destination)
        self._save_sigmoid(destination)
        expander = TemplateExpander(self._config)
        destination.create_subpath("lstm_cell").as_file(".vhd").write_text(
            expander.lines()
        )

    def _save_sigmoid(self, destination: Path):
        sigmoid_destination = destination.create_subpath("hard_sigmoid")
        sigmoid = HardSigmoid(
            width=self.total_bits,
            lower_bound_for_zero=self._fp_config.as_integer(-3),
            upper_bound_for_one=self._fp_config.as_integer(3),
        )
        sigmoid.save_to(sigmoid_destination)

    def _save_hardtanh(self, destination: Path):
        hardtanh_destination = destination.create_subpath("hard_tanh")
        hardtanh = FPHardTanh(
            total_bits=self.total_bits, frac_bits=self._fp_config.frac_bits
        )
        hardtanh.save_to(hardtanh_destination)

    def _write_files(
        self,
        destination: Path,
        rom_template: TemplateExpander,
        names: tuple[str, ...],
        parameters: Any,
        address_width: int,
    ):
        for values, name in zip(parameters, names):
            self._update_rom_base_config(
                name=name, values=values, address_width=address_width
            )
            rom_file = destination.create_subpath(f"{name}_rom").as_file(".vhd")
            rom_file.write_text(rom_template.lines())

    def _update_rom_base_config(self, values: list[int], name: str, address_width: int):
        values = self._pad_with_zeros(values, address_width)
        self._rom_base_config.parameters.update(
            dict(
                resource_option="auto",
                name=f"rom_{name}_{self.name}",
                rom_addr_bitwidth=str(address_width),
                rom_data_bitwidth=str(self.total_bits),
                rom_value=",".join(map(self._to_hex, values)),
            )
        )

    @staticmethod
    def _to_hex(value):
        return generate_hex_for_rom(to_hex(value, bit_width=8))

    @staticmethod
    def _pad_with_zeros(values, address_width):
        suffix = list(repeat(0, 2**address_width - len(values)))
        return values + suffix

    def _save_dual_port_double_clock_ram(self, destination: Path):
        template_configuration = TemplateConfig(
            file_name="dual_port_2_clock_ram.tpl.vhd",
            package=module_to_package(self.__module__),
            parameters=dict(
                name=f"{self.name}_dual_port_2_clock_ram",
            ),
        )
        template_expansion = TemplateExpander(template_configuration)

        destination.create_subpath(f"{self.name}_dual_port_2_clock_ram").as_file(
            ".vhd"
        ).write_text(template_expansion.lines())
