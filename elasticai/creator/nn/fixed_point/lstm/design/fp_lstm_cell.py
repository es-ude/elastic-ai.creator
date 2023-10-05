from collections.abc import Iterable
from functools import partial
from typing import Any, cast

import numpy as np

from ._common_imports import (
    Design,
    FixedPointConfig,
    InProjectTemplate,
    Path,
    Port,
    Rom,
    Signal,
    calculate_address_width,
    module_to_package,
    std_signals,
)


class FPLSTMCell(Design):
    def __init__(
        self,
        *,
        name: str,
        hardtanh: Design,
        hardsigmoid: Design,
        total_bits: int,
        frac_bits: int,
        w_ih: list[list[list[int]]],
        w_hh: list[list[list[int]]],
        b_ih: list[list[int]],
        b_hh: list[list[int]],
    ) -> None:
        super().__init__(name=name)
        work_library_name: str = "work"

        self.input_size = len(w_ih[0])
        self.hidden_size = len(w_ih) // 4
        self.weights_ih = w_ih
        self.weights_hh = w_hh
        self.biases_ih = b_ih
        self.biases_hh = b_hh
        self._config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
        self._htanh = hardtanh
        self._hsigmoid = hardsigmoid
        self._template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name=f"{self.name}.tpl.vhd",
            parameters=dict(
                name=self.name,
                library=work_library_name,
                tanh_name=self._htanh.name,
                sigmoid_name=self._hsigmoid.name,
                data_width=str(total_bits),
                frac_width=str(frac_bits),
                input_size=str(self.input_size),
                hidden_size=str(self.hidden_size),
                x_h_addr_width=str(
                    calculate_address_width(self.input_size + self.hidden_size)
                ),
                hidden_addr_width=str(calculate_address_width(self.hidden_size)),
                w_addr_width=str(
                    calculate_address_width(
                        (self.input_size + self.hidden_size) * self.hidden_size
                    )
                ),
            ),
        )

    @property
    def total_bits(self) -> int:
        return int(cast(str, self._template.parameters["data_width"]))

    @property
    def frac_bits(self) -> int:
        return int(cast(str, self._template.parameters["frac_width"]))

    @property
    def _hidden_addr_width(self) -> int:
        return int(cast(str, self._template.parameters["hidden_addr_width"]))

    @property
    def _weight_address_width(self) -> int:
        return int(cast(str, self._template.parameters["w_addr_width"]))

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

        destination.create_subpath("lstm_cell").as_file(".vhd").write(self._template)

    def _build_weights(self) -> tuple[list[list], list[list]]:
        weights = np.concatenate(
            (np.array(self.weights_ih), np.array(self.weights_hh)), axis=1
        )
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

    def _save_hardtanh(self, destination: Path) -> None:
        self._htanh.save_to(destination)

    def _save_sigmoid(self, destination: Path) -> None:
        self._hsigmoid.save_to(destination)

    def _save_dual_port_double_clock_ram(self, destination: Path) -> None:
        template = InProjectTemplate(
            file_name="dual_port_2_clock_ram.tpl.vhd",
            package=module_to_package(self.__module__),
            parameters=dict(name=self.name),
        )
        name = f"dual_port_2_clock_ram_{self.name}"
        destination.create_subpath(name).as_file(".vhd").write(template)
