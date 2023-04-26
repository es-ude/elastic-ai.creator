import torch
from torch import Tensor
from torch.nn import Tanh as _BaseTanh

from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.hdl.translatable import Translatable
from elasticai.creator.hdl.vhdl.designs.monotonously_increasing_precomputed_scalar_function.precomputed_scalar_function import (
    _PrecomputedMonotonouslyIncreasingScalarFunction,
)
from elasticai.creator.nn.fixed_point_arithmetics import FixedPointArithmetics
from elasticai.creator.nn.two_complement_fixed_point_config import FixedPointConfig


class PrecomputedTanh(_BaseTanh, Translatable):
    def __init__(self, total_bits: int, frac_bits: int):
        super().__init__()
        self.op = FixedPointArithmetics(
            config=FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
        )
        self._total_bits = total_bits
        self._precomputed_values: list = list()

    def forward(self, input: Tensor) -> Tensor:
        result = super().forward(input)
        result = self.op.quantize(result)
        return result

    def _sampling_function(self, input: int) -> int:
        return self._precomputed_values[input]

    def translate(self, name: str) -> Design:
        training_state = self.training
        self.train(False)
        self._precomputed_values = self.forward(
            torch.tensor(tuple(range(2**self._total_bits)))
        ).tolist()
        self.train(training_state)
        return _PrecomputedMonotonouslyIncreasingScalarFunction(
            name=name,
            width=self._total_bits,
            inputs=range(2**self._total_bits),
            function=self._sampling_function,
        )
