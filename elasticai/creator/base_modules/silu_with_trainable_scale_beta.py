import torch

from elasticai.creator.base_modules.math_operations import Quantize


class SiLUWithTrainableScaleBeta(torch.nn.SiLU):
    def __init__(self, operations: Quantize) -> None:
        super().__init__(inplace=False)
        self._operations = operations
        self.scale = torch.nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = torch.nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        a = self.scale * super().forward(input) + self.beta
        x = self._operations.quantize(a)
        return x
