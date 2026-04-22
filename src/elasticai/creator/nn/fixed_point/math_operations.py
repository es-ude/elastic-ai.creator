from typing import cast

import torch

from elasticai.creator.arithmetic import FxpArithmetic
from elasticai.creator.base_modules.conv1d import MathOperations as Conv1dOps
from elasticai.creator.base_modules.linear import MathOperations as LinearOps
from elasticai.creator.base_modules.lstm_cell import MathOperations as LSTMOps
from elasticai.creator.nn.fixed_point.fxp_round_cut import (
    CutToFixedPoint,
    RoundToFixedPoint,
)


class MathOperations(LinearOps, Conv1dOps, LSTMOps):
    def __init__(self, config: FxpArithmetic) -> None:
        self.config = config

    def _clamp(self, a: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            input=a,
            min=self.config.minimum_as_rational,
            max=self.config.maximum_as_rational,
        )

    def _cut(self, a: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, CutToFixedPoint.apply(a, self.config))

    def _round(self, a: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, RoundToFixedPoint.apply(a, self.config))

    def round(self, a: torch.Tensor) -> torch.Tensor:
        return self._round(self._clamp(a))

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._clamp(torch.add(a, b))

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(torch.matmul(a, b))

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(torch.mul(a, b))

    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return self._cut(self._clamp(a))
