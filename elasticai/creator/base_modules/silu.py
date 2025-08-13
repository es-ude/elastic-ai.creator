import torch


class SiLU(torch.nn.SiLU):
    def __init__(self) -> None:
        super().__init__(inplace=False)
