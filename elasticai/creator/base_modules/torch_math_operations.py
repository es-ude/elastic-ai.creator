import torch

from .math_operations import Add, MatMul, Mul, Quantize


class TorchMathOperations(Quantize, Add, MatMul, Mul):
    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return a

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)
