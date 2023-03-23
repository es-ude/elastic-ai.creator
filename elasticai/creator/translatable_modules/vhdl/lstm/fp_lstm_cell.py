from copy import copy
from functools import partial
from typing import Any, cast

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
from elasticai.creator.hdl.vhdl.code_generation.twos_complement import to_unsigned
from elasticai.creator.hdl.vhdl.designs import HardSigmoid
from elasticai.creator.hdl.vhdl.designs.rom import Rom
from elasticai.creator.translatable_modules.vhdl.lstm.fp_hard_tanh import FPHardTanh


class FPLSTMCell(Design):
    def __init__(
        self,
        *,
        name: str,
        total_bits: int,
        frac_bits: int,
        lower_bound_for_hard_sigmoid: int,
        upper_bound_for_hard_sigmoid: int,
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
        self._upper_bound_for_hard_sigmoid = upper_bound_for_hard_sigmoid
        self._lower_bound_for_hard_sigmoid = lower_bound_for_hard_sigmoid
        self._rom_base_config = copy(base_config)
        self._rom_base_config.file_name = "rom.tpl.vhd"

        self._config = copy(base_config)
        self._config.file_name = f"{self.name}.tpl.vhd"
        self._config.parameters = {
            k: str(v)
            for k, v in dict(
                name=self.name,
                library="work",
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
    def frac_bits(self) -> int:
        return int(cast(str, self._config.parameters["frac_width"]))

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

        final_weights = (w_i, w_f, w_g, w_o)
        final_biases = (b_i, b_f, b_g, b_o)

        return final_weights, final_biases

    def save_to(self, destination: "Path"):
        weights, biases = self._build_weights()

        write_files = partial(self._write_files, destination=destination)
        write_files(
            names=("wi", "wf", "wg", "wo"),
            parameters=weights,
        )
        write_files(
            names=("bi", "bf", "bg", "bo"),
            parameters=biases,
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
            lower_bound_for_zero=to_unsigned(
                self._lower_bound_for_hard_sigmoid, total_bits=self.total_bits
            ),
            upper_bound_for_one=to_unsigned(
                self._upper_bound_for_hard_sigmoid, total_bits=self.total_bits
            ),
        )
        sigmoid.save_to(sigmoid_destination)

    def _save_hardtanh(self, destination: Path):
        hardtanh_destination = destination.create_subpath("hard_tanh")
        hardtanh = FPHardTanh(total_bits=self.total_bits, frac_bits=self.frac_bits)
        hardtanh.save_to(hardtanh_destination)

    def _write_files(
        self,
        destination: Path,
        names: tuple[str, ...],
        parameters: Any,
    ):
        for values, name in zip(parameters, names):
            rom = Rom(
                name=f"rom_{name}_lstm_cell",
                values_as_integers=values,
                data_width=self.total_bits,
            )
            rom.save_to(destination.create_subpath(f"{name}_rom"))

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
