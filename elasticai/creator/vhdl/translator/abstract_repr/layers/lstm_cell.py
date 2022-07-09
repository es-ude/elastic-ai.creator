from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from elasticai.creator.vhdl.components.dual_port_2_clock_ram import DualPort2ClockRam
from elasticai.creator.vhdl.components.lstm_cell import LSTMCell as LSTMCellVHDL
from elasticai.creator.vhdl.components.lstm_common import LSTMCommon
from elasticai.creator.vhdl.components.rom import Rom
from elasticai.creator.vhdl.components.sigmoid import Sigmoid
from elasticai.creator.vhdl.components.tanh import Tanh
from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.number_representations import FixedPoint


@dataclass
class LSTMCell:
    weights_ii: list[list[FixedPoint]]
    weights_hi: list[list[FixedPoint]]
    weights_if: list[list[FixedPoint]]
    weights_hf: list[list[FixedPoint]]
    weights_ig: list[list[FixedPoint]]
    weights_hg: list[list[FixedPoint]]
    weights_io: list[list[FixedPoint]]
    weights_ho: list[list[FixedPoint]]

    bias_ii: list[FixedPoint]
    bias_hi: list[FixedPoint]
    bias_if: list[FixedPoint]
    bias_hf: list[FixedPoint]
    bias_ig: list[FixedPoint]
    bias_hg: list[FixedPoint]
    bias_io: list[FixedPoint]
    bias_ho: list[FixedPoint]

    def _build_weights(
        self,
    ) -> tuple[tuple[list[FixedPoint], ...], tuple[list[FixedPoint], ...]]:
        def concat_weights(a, b) -> list[FixedPoint]:
            return np.hstack((a, b)).flatten().tolist()  # type: ignore

        w_i = concat_weights(self.weights_ii, self.weights_hi)
        w_f = concat_weights(self.weights_if, self.weights_hf)
        w_g = concat_weights(self.weights_ig, self.weights_hg)
        w_o = concat_weights(self.weights_io, self.weights_ho)

        b_i = self.bias_ii + self.bias_hi
        b_f = self.bias_if + self.bias_hf
        b_g = self.bias_ig + self.bias_hg
        b_o = self.bias_io + self.bias_ho

        return (w_i, w_f, w_g, w_o), (b_i, b_f, b_g, b_o)

    def translate(
        self,
        fixed_point_factory: Callable[[float | int], FixedPoint],
        sigmoid_linspace_args: tuple[float, float, int],
        tanh_linspace_args: tuple[float, float, int],
    ) -> Iterator[tuple[str, Code]]:
        def to_fp(values: list[float | int]) -> list[FixedPoint]:
            return list(map(fixed_point_factory, values))

        weights, bias = self._build_weights()
        rom_names = (
            f"{name}_rom" for name in ("wi", "wf", "wg", "wo", "bi", "bf", "bg", "bo")
        )
        for rom_values, rom_name in zip(weights + bias, rom_names):
            rom = Rom(rom_name=rom_name, values=rom_values, resource_option="auto")
            yield rom.file_name, list(rom())

        sigmoid = Sigmoid(
            x=to_fp(np.linspace(*sigmoid_linspace_args).tolist()),  # type: ignore
            component_name="sigmoid",
        )
        yield sigmoid.file_name, list(sigmoid())

        tanh = Tanh(
            x=to_fp(np.linspace(*tanh_linspace_args).tolist()),  # type: ignore
            component_name="tanh",
        )
        yield tanh.file_name, list(tanh())

        for static_cls in (LSTMCellVHDL, LSTMCommon, DualPort2ClockRam):
            static_obj = static_cls()
            yield static_obj.file_name, list(static_obj())
