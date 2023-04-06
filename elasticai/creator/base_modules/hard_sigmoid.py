import torch


class HardSigmoid(torch.nn.Hardsigmoid):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)
