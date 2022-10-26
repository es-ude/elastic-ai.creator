import torch
import torch.nn.functional as F

from elasticai.creator.vhdl.number_representations import (
    FixedPointFactory,
    fixed_point_params_from_factory,
)
from elasticai.creator.vhdl.quantized_modules._typing.quant_type import QuantType
from elasticai.creator.vhdl.quantized_modules.autograd_functions import (
    FixedPointDequantFunction,
    FixedPointQuantFunction,
)


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
        self.fixed_point_factory = fixed_point_factory

    def quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        _, frac_bits = fixed_point_params_from_factory(self.fixed_point_factory)

        def fp(value: float) -> int:
            return int(value * (1 << frac_bits))

        def fp_hard_sigmoid(a: int) -> int:
            if a <= fp(-3):
                return 0
            elif a >= fp(3):
                return fp(1)
            else:
                return int(a * fp(1 / 6) / fp(1)) + fp(1 / 2)

        return x.apply_(fp_hard_sigmoid)
