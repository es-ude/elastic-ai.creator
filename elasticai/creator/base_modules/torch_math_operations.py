import torch

from .conv1d import MathOperations as Conv1dOps
from .linear import MathOperations as LinearOps
from .lstm_cell import MathOperations as LSTMOps
from .silu_with_trainable_scale_beta import MathOperations as SiluOps


class TorchMathOperations(LinearOps, Conv1dOps, LSTMOps, SiluOps):
    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return a

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)
