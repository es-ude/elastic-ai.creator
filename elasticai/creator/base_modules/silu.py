import torch

from elasticai.creator.base_modules.arithmetics.arithmetics import Arithmetics


class SiLU(torch.nn.SiLU):
    def __init__(self, arithmetics: Arithmetics) -> None:
        super().__init__(inplace=False)
        self._arithmetics = arithmetics
        self.scale = torch.nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = torch.nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        a = self.scale * super().forward(input) + self.beta
        x = self._arithmetics.quantize(a)
        return x
