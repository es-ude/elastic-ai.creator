from collections.abc import Iterable
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
from elasticai.creator.hdl.design_base import std_signals
from elasticai.creator.hdl.design_base.design import Design, Port
from elasticai.creator.hdl.design_base.signal import Signal
from elasticai.creator.hdl.translatable import Path
from elasticai.creator.hdl.vhdl.code_generation.twos_complement import to_unsigned
from elasticai.creator.hdl.vhdl.designs import HardSigmoid
from elasticai.creator.hdl.vhdl.designs.rom import Rom
from elasticai.creator.nn.vhdl.lstm.design.fp_hard_tanh import FPHardTanh


class FPLSTMCell(Design):
    def __init__(
        self,
        *,
        name: str,
        total_bits: int,
        frac_bits: int,
        lower_bound_for_hard_sigmoid: int,
        upper_bound_for_hard_sigmoid: int,
        w_ih: list[list[list[int]]],
        w_hh: list[list[list[int]]],
        b_ih: list[list[int]],
        b_hh: list[list[int]],
        work_library_name: str = "work",
    ) -> None:
        super().__init__(name=name)
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
                library=work_library_name,
                data_width=total_bits,
                frac_width=frac_bits,
                input_size=self.input_size,
                hidden_size=self.hidden_size,
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
                std_signals.clock(),
                # ctrl_signal("clk_hadamard"),
                ctrl_signal("reset"),
                std_signals.enable(),
                ctrl_signal("zero_state"),
                Signal("x_data", width=self.total_bits),
                ctrl_signal("h_out_en"),
                Signal("h_out_addr", width=self._hidden_addr_width),
            ],
            outgoing=[
                std_signals.done(),
                Signal("h_out_data", self.total_bits),
            ],
        )

    def save_to(self, destination: Path) -> None:
        weights, biases = self._build_weights()

        self._save_roms(
            destination=destination,
            names=("wi", "wf", "wg", "wo", "bi", "bf", "bg", "bo"),
            parameters=[*weights, *biases],
        )
        self._save_dual_port_double_clock_ram(destination)
        self._save_hardtanh(destination)
        self._save_sigmoid(destination)

        expander = TemplateExpander(self._config)
        destination.create_subpath("lstm_cell").as_file(".vhd").write_text(
            expander.lines()
        )

    def _build_weights(self) -> tuple[list[list], list[list]]:
        weights = np.concatenate((self.weights_ih, self.weights_hh), axis=1)
        w_i, w_f, w_g, w_o = weights.reshape(4, -1).tolist()

        bias = np.add(self.biases_ih, self.biases_hh)
        b_i, b_f, b_g, b_o = bias.reshape(4, -1).tolist()

        return [w_i, w_f, w_g, w_o], [b_i, b_f, b_g, b_o]

    def _save_roms(
        self, destination: Path, names: Iterable[str], parameters: Iterable[Any]
    ) -> None:
        suffix = f"_rom_{self.name}"
        for name, values in zip(names, parameters):
            rom = Rom(
                name=name + suffix,
                data_width=self.total_bits,
                values_as_integers=values,
            )
            rom.save_to(destination.create_subpath(name + suffix))

    def _save_sigmoid(self, destination: Path) -> None:
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

    def _save_hardtanh(self, destination: Path) -> None:
        hardtanh_destination = destination.create_subpath("hard_tanh")
        hardtanh = FPHardTanh(total_bits=self.total_bits, frac_bits=self.frac_bits)
        hardtanh.save_to(hardtanh_destination)

    def _save_dual_port_double_clock_ram(self, destination: Path) -> None:
        template_configuration = TemplateConfig(
            file_name="dual_port_2_clock_ram.tpl.vhd",
            package=module_to_package(self.__module__),
            parameters=dict(name=self.name),
        )
        template_expansion = TemplateExpander(template_configuration)

        destination.create_subpath(f"dual_port_2_clock_ram_{self.name}").as_file(
            ".vhd"
        ).write_text(template_expansion.lines())
