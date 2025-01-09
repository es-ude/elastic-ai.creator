import torch
from torch.nn import Sigmoid as _Sigmoid

from .. import MathOperations


class Sigmoid(_Sigmoid):
    def __init__(self, operations: MathOperations):
        super().__init__()
        self._operations = operations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._operations.quantize(super().forward(x))
