import torch
import torch.nn.functional as F

from elasticai.creator.vhdl.number_representations import FixedPointFactory
from elasticai.creator.vhdl.quantized_modules.autograd_functions import (
    FixedPointDequantFunction,
    FixedPointQuantFunction,
)
from elasticai.creator.vhdl.quantized_modules.typing import QuantType


class _ReLUBase(torch.nn.ReLU):
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
        return F.relu(q_x)

    def quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("The quantized_forward function is not implemented.")


class FixedPointReLU(_ReLUBase):
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

    def quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)
