import torch


class AdaptableSiLU(torch.nn.SiLU):
    def __init__(self) -> None:
        super().__init__(inplace=False)
        self.scale = torch.nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = torch.nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.scale * super().forward(input) + self.beta
