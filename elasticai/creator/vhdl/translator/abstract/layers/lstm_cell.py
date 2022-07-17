from dataclasses import dataclass
from typing import Callable

import numpy as np

from elasticai.creator.vhdl.components import (
    DualPort2ClockRam,
    LSTMCommon,
    Rom,
    Sigmoid,
    Tanh,
)
from elasticai.creator.vhdl.components.lstm_cell import LSTMCell as LSTMCellVHDL
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.vhdl_component import VHDLModule


@dataclass
class LSTMCell:
    weights_ii: list[list[float]]
    weights_hi: list[list[float]]
    weights_if: list[list[float]]
    weights_hf: list[list[float]]
    weights_ig: list[list[float]]
    weights_hg: list[list[float]]
    weights_io: list[list[float]]
    weights_ho: list[list[float]]

    bias_ii: list[float]
    bias_hi: list[float]
    bias_if: list[float]
    bias_hf: list[float]
    bias_ig: list[float]
    bias_hg: list[float]
    bias_io: list[float]
    bias_ho: list[float]

    def _build_weights(
        self, to_fixed_point: Callable[[list[float]], list[FixedPoint]]
    ) -> tuple[tuple[list[FixedPoint], ...], tuple[list[FixedPoint], ...]]:
        def concat_weights(a, b) -> list[float]:
            return np.hstack((a, b)).flatten().tolist()  # type: ignore

        w_i = concat_weights(self.weights_ii, self.weights_hi)
        w_f = concat_weights(self.weights_if, self.weights_hf)
        w_g = concat_weights(self.weights_ig, self.weights_hg)
        w_o = concat_weights(self.weights_io, self.weights_ho)

        b_i = self.bias_ii + self.bias_hi
        b_f = self.bias_if + self.bias_hf
        b_g = self.bias_ig + self.bias_hg
        b_o = self.bias_io + self.bias_ho

        weights = tuple(map(to_fixed_point, (w_i, w_f, w_g, w_o)))
        bias = tuple(map(to_fixed_point, (b_i, b_f, b_g, b_o)))

        return weights, bias

    def translate(
        self,
        fixed_point_factory: Callable[[float], FixedPoint],
        sigmoid_linspace_args: tuple[float, float, int],
        tanh_linspace_args: tuple[float, float, int],
    ) -> VHDLModule:
        def to_fp(values: list[float]) -> list[FixedPoint]:
            return list(map(fixed_point_factory, values))

        weights, bias = self._build_weights(to_fixed_point=to_fp)
        rom_names = (
            f"{name}_rom" for name in ("wi", "wf", "wg", "wo", "bi", "bf", "bg", "bo")
        )
        for rom_values, rom_name in zip(weights + bias, rom_names):
            yield Rom(rom_name=rom_name, values=rom_values, resource_option="auto")

        yield Sigmoid(
            x=to_fp(np.linspace(*sigmoid_linspace_args).tolist()),  # type: ignore
            component_name="sigmoid",
        )

        yield Tanh(
            x=to_fp(np.linspace(*tanh_linspace_args).tolist()),  # type: ignore
            component_name="tanh",
        )

        for static_cls in (LSTMCellVHDL, LSTMCommon, DualPort2ClockRam):
            yield static_cls()
