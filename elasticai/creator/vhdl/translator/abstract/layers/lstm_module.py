from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

from elasticai.creator.vhdl.components import (
    DualPort2ClockRamComponent,
    LSTMCommonComponent,
    LSTMComponent,
    RomComponent,
    SigmoidComponent,
    TanhComponent,
)
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.vhdl_component import VHDLComponent, VHDLModule


@dataclass
class LSTMTranslationArgs:
    fixed_point_factory: Callable[[float], FixedPoint]
    sigmoid_resolution: tuple[float, float, int]
    tanh_resolution: tuple[float, float, int]
    work_library_name: str = field(default="work")


@dataclass
class LSTMModule(VHDLModule):
    """
    Abstract representation of an LSTM layer that can be directly translated to an iterable of VHDLComponent objects.
    Currently, no stacked LSTMs are supported (only single layer LSTMs are supported).

    Parameters:
        weights_ih (list[list[list[float]]]):
            List of input-hidden weights for each layer. Weights of one layer is a list[list[float]] with the shape
            (4*hidden_size, input_size) and the structure (W_ii | W_if | W_ig | W_io).
        weights_hh (list[list[list[float]]]):
            List of hidden-hidden weights for each layer. Weights of one layer is a list[list[float]] with the shape
            (4*hidden_size, hidden_size) and the structure (W_hi | W_hf | W_hg | W_ho).
        biases_ih (list[list[float]]):
            List of input-hidden biases for each layer. Biases of one layer is a list[float] with the shape
            (4*hidden_size,) and the structure (b_ii | b_if | b_ig | b_io).
        biases_hh (list[list[float]]):
            List of hidden-hidden biases for each layer. Biases of one layer is a list[float] with the shape
            (4*hidden_size,) and the structure (b_hi | b_hf | b_hg | b_ho).
    """

    weights_ih: list[list[list[float]]]
    weights_hh: list[list[list[float]]]
    biases_ih: list[list[float]]
    biases_hh: list[list[float]]

    def _build_weights(
        self, to_fixed_point: Callable[[list[float]], list[FixedPoint]]
    ) -> tuple[tuple[list[FixedPoint], ...], ...]:
        def reshape_params(params: list[list[float]] | np.ndarray) -> np.ndarray:
            return np.array([np.reshape(param, (4, -1)) for param in params])

        weights = np.concatenate((self.weights_ih, self.weights_hh), axis=2)
        weights = reshape_params(weights)
        weights = weights[0]  # Currently only supporting one layer LSTMs
        w_i, w_f, w_g, w_o = weights.reshape(4, -1).tolist()

        bias = np.add(self.biases_ih, self.biases_hh)
        bias = reshape_params(bias)
        bias = bias[0]  # Currently only supporting one layer LSTMs
        b_i, b_f, b_g, b_o = bias.tolist()

        final_weights = tuple(map(to_fixed_point, (w_i, w_f, w_g, w_o)))
        final_biases = tuple(map(to_fixed_point, (b_i, b_f, b_g, b_o)))

        return final_weights, final_biases

    def _derive_input_and_hidden_size(self) -> tuple[int, int]:
        _, hidden_size, input_size = np.shape(self.weights_ih)
        return input_size, hidden_size // 4

    def components(self, args: LSTMTranslationArgs) -> Iterator[VHDLComponent]:
        def to_fp(values: list[float]) -> list[FixedPoint]:
            return list(map(args.fixed_point_factory, values))

        weights, bias = self._build_weights(to_fixed_point=to_fp)
        rom_names = (
            f"{name}_rom" for name in ("wi", "wf", "wg", "wo", "bi", "bf", "bg", "bo")
        )
        for rom_values, rom_name in zip(weights + bias, rom_names):
            yield RomComponent(
                rom_name=rom_name, values=rom_values, resource_option="auto"
            )

        precomputed_sigmoid_inputs = to_fp(np.linspace(*args.sigmoid_resolution).tolist())  # type: ignore
        precomputed_tanh_inputs = to_fp(np.linspace(*args.tanh_resolution).tolist())  # type: ignore
        yield SigmoidComponent(x=precomputed_sigmoid_inputs)
        yield TanhComponent(x=precomputed_tanh_inputs)

        input_size, hidden_size = self._derive_input_and_hidden_size()
        yield LSTMComponent(
            input_size=input_size,
            hidden_size=hidden_size,
            fixed_point_factory=args.fixed_point_factory,
            work_library_name=args.work_library_name,
        )

        yield LSTMCommonComponent()
        yield DualPort2ClockRamComponent()
