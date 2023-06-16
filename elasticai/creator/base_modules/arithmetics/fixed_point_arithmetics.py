from typing import Optional, cast

import torch

from elasticai.creator.base_modules.autograd_functions.round_to_fixed_point import (
    RoundToFixedPoint,
)
from elasticai.creator.base_modules.two_complement_fixed_point_config import (
    FixedPointConfig,
)

from .arithmetics import Arithmetics


class FixedPointArithmetics(Arithmetics):
    def __init__(self, config: FixedPointConfig) -> None:
        self.config = config

    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return self.round(self.clamp(a))

    def clamp(self, a: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            a, min=self.config.minimum_as_rational, max=self.config.maximum_as_rational
        )

    def round(self, a: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, RoundToFixedPoint.apply(a, self.config))

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.clamp(a + b)

    def sum(
        self, a: torch.Tensor, dim: Optional[int | tuple[int, ...]] = None
    ) -> torch.Tensor:
        return self.clamp(torch.sum(a, dim=dim))

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(a * b)

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(torch.matmul(a, b))

    def conv1d(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor | None,
        stride: int,
        padding: int | str,
        dilation: int,
        groups: int,
    ) -> torch.Tensor:
        outputs = torch.nn.functional.conv1d(
            input=inputs,
            weight=weights,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        return self.quantize(outputs)
