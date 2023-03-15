from typing import Any, cast

import torch

from elasticai.creator.nn._two_complement_fixed_point_config import FixedPointConfig
from elasticai.creator.nn.arithmetics import Arithmetics
from elasticai.creator.nn.fixed_point_arithmetics import FixedPointArithmetics


class Linear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        arithmetics: Arithmetics,
        bias: bool,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.ops = arithmetics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.ops.quantize(self.weight)

        if self.bias is not None:
            bias = self.ops.quantize(self.bias)
            return self.ops.add(self.ops.matmul(x, weight.T), bias)

        return self.ops.matmul(x, weight.T)


class FixedPointLinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        total_bits: int,
        frac_bits: int,
        bias: bool,
        device: Any = None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            arithmetics=FixedPointArithmetics(
                config=FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
            ),
            bias=bias,
            device=device,
        )

    @property
    def total_bits(self) -> int:
        return cast(FixedPointArithmetics, self.ops).config.total_bits

    @property
    def frac_bits(self) -> int:
        return cast(FixedPointArithmetics, self.ops).config.frac_bits

    @property
    def fixed_point_factory(self) -> FixedPointConfig:
        return cast(FixedPointArithmetics, self.ops).config
