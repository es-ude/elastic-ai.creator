from abc import abstractmethod

import torch
from torch import Tensor

from elasticai.creator.base_modules.conv1d import MathOperations as Conv1dOps
from elasticai.creator.base_modules.linear import MathOperations as LinearOps
from elasticai.creator.base_modules.lstm_cell import MathOperations as LSTMOps


class MathOperations(LinearOps, LSTMOps, Conv1dOps):
    @abstractmethod
    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(a + b)

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(torch.matmul(a, b))

    def mul(self, a: Tensor, b: Tensor) -> Tensor:
        return self.quantize(torch.mul(a, b))

    def div(self, a: Tensor, b: Tensor) -> Tensor:
        return self.quantize(torch.div(a, b))
