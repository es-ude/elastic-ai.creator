import torch

from .conv1d import MathOperations as Conv1dOps
from .linear import MathOperations as LinearOps
from .lstm_cell import MathOperations as LSTMOps


class TorchMathOperations(LinearOps, Conv1dOps, LSTMOps):
    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return a

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.add(a, b)

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mul(a, b)

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)
