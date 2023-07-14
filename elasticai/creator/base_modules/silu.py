import torch

from elasticai.creator.base_modules.arithmetics.arithmetics import Arithmetics


class SiLU(torch.nn.SiLU):
    def __init__(self, arithmetics: Arithmetics) -> None:
        super().__init__(inplace=False)
        self._arithmetics = arithmetics
        self.beta = torch.nn.Parameter(torch.ones(1, requires_grad=True))
        self.asymetric_offset = torch.nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        a = super().forward(self.beta * input - self.asymetric_offset)
        x = self._arithmetics.quantize(a)
        return x
