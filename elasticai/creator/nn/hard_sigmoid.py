import torch
import torch.nn.functional as F

from elasticai.creator.nn.autograd_functions.fixed_point_quantization import (
    FixedPointDequantFunction,
    FixedPointQuantFunction,
)
from elasticai.creator.nn.quant_typings import QuantType
from elasticai.creator.vhdl.number_representations import FixedPointFactory


class _HardSigmoidBase(torch.nn.Hardsigmoid):
    def __init__(
        self,
        input_quant: QuantType = lambda x: x,
        input_dequant: QuantType = lambda x: x,
        inplace: bool = False,
    ) -> None:
        super().__init__(inplace)
        self.input_quant = input_quant
        self.input_dequant = input_dequant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_x = self.input_dequant(self.input_quant(x))
        return F.hardsigmoid(q_x)

    def quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("The quantized_forward function is not implemented.")


class FixedPointHardSigmoid(_HardSigmoidBase):
    def __init__(
        self, fixed_point_factory: FixedPointFactory, inplace: bool = False
    ) -> None:
        super().__init__(
            input_quant=lambda x: FixedPointQuantFunction.apply(x, fixed_point_factory),
            input_dequant=lambda x: FixedPointDequantFunction.apply(
                x, fixed_point_factory
            ),
            inplace=inplace,
        )
        self.fp_factory = fixed_point_factory

    def quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        def fp(value: float) -> int:
            largest_fp_int = 2 ** (self.fp_factory.total_bits - 1) - 1
            int_value = int(value * (1 << self.fp_factory.frac_bits))
            return min(max(int_value, -largest_fp_int), largest_fp_int)

        def fp_hard_sigmoid(a: int) -> int:
            if a <= fp(-3.0):
                return 0
            elif a >= fp(3.0):
                return fp(1.0)
            else:
                return int(a * fp(1.0 / 6.0) / fp(1.0)) + fp(0.5)

        return x.detach().apply_(fp_hard_sigmoid)
