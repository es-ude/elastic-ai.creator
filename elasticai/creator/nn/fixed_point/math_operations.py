from typing import cast

import torch
from torch import Tensor

from elasticai.creator.base_modules.conv1d import MathOperations as Conv1dOps
from elasticai.creator.base_modules.linear import MathOperations as LinearOps
from elasticai.creator.base_modules.lstm_cell import MathOperations as LSTMOps

from .round_to_fixed_point import RoundToFixedPoint
from .two_complement_fixed_point_config import FixedPointConfig


class MathOperations(LinearOps, Conv1dOps, LSTMOps):
    def __init__(self, config: FixedPointConfig) -> None:
        self.config = config

    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return self._round(self._clamp(a))

    def _clamp(self, a: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            a, min=self.config.minimum_as_rational, max=self.config.maximum_as_rational
        )

    def _round(self, a: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, RoundToFixedPoint.apply(a, self.config))

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._clamp(a + b)

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(torch.matmul(a, b))

    def mul(self, a: Tensor, b: Tensor) -> Tensor:
        return self.quantize(a * b)
