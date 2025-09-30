import torch
from torch import Tensor

class LinearQuantizationMathOps:
    def quantize(self, a: Tensor) -> Tensor:
        return a

    def add(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.add(a, b)

    def mul(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.mul(a, b)

    def matmul(self, a: Tensor, a_scale:Tensor, a_zero_point: Tensor, b: Tensor, b_scale: Tensor, b_zero_point: Tensor) -> Tensor:

        return torch.matmul(a, b)
