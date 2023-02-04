from typing import Any, cast

import torch

from elasticai.creator.nn.arithmetics import Arithmetics, FixedPointArithmetics
from elasticai.creator.vhdl.number_representations import FixedPointFactory


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
        fixed_point_factory: FixedPointFactory,
        bias: bool,
        device: Any = None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            arithmetics=FixedPointArithmetics(fixed_point_factory=fixed_point_factory),
            bias=bias,
            device=device,
        )

    @property
    def fixed_point_factory(self) -> FixedPointFactory:
        return cast(FixedPointArithmetics, self.ops).fixed_point_factory
