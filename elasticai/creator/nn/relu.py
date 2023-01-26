import torch


class ReLU(torch.nn.ReLU):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)
