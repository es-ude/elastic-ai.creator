from typing import cast

import torch
from torch import Tensor

from elasticai.creator.base_modules.conv1d import MathOperations as Conv1dOps
from elasticai.creator.base_modules.linear import MathOperations as LinearOps
from elasticai.creator.base_modules.lstm_cell import MathOperations as LSTMOps

from ._round_to_fixed_point_autograd import RoundToFixedPoint
from ._two_complement_fixed_point_config import FixedPointConfigV2


class MathOperations(LinearOps, Conv1dOps, LSTMOps):
    def __init__(
        self,
        forward_config: FixedPointConfigV2,
        backward_config: FixedPointConfigV2 | None = None,
    ) -> None:
        self.config = forward_config
        self.grad_config = backward_config

    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            RoundToFixedPoint.apply(a, self.config, self.grad_config),
        )

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(a + b)

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(torch.matmul(a, b))

    def mul(self, a: Tensor, b: Tensor) -> Tensor:
        return self.quantize(a * b)
