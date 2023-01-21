import torch

from elasticai.creator.nn.arithmetics import Arithmetics, FixedPointArithmetics
from elasticai.creator.nn.autograd_functions.fixed_point_quantization import (
    FixedPointDequantFunction,
    FixedPointQuantFunction,
)
from elasticai.creator.nn.quantization import FakeQuant, QuantType
from elasticai.creator.vhdl.number_representations import FixedPointFactory


class _LinearBase(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        arithmetics: Arithmetics,
        bias: bool = True,
        input_quant: QuantType = lambda x: x,
        param_quant: QuantType = lambda x: x,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.ops = arithmetics
        self.input_quant = input_quant
        self.param_quant = param_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        weight = self.param_quant(self.weight)

        if self.bias is not None:
            bias = self.param_quant(self.bias)
            return self.ops.add(self.ops.matmul(x, weight.T), bias)

        return self.ops.matmul(x, weight.T)


class FixedPointLinear(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        fixed_point_factory: FixedPointFactory,
        bias: bool = True,
        device=None,
    ) -> None:
        self.arithmetics = FixedPointArithmetics(
            fixed_point_factory=fixed_point_factory
        )
        self.quant = FakeQuant(
            quant=lambda x: FixedPointQuantFunction.apply(x, fixed_point_factory),
            dequant=lambda x: FixedPointDequantFunction.apply(x, fixed_point_factory),
            arithmetics=self.arithmetics,
        )
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            arithmetics=self.arithmetics,
            input_quant=self.quant,
            param_quant=self.quant,
            device=device,
        )
