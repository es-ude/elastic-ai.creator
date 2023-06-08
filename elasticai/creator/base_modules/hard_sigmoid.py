import torch


class HardSigmoid(torch.nn.Hardsigmoid):
    def __init__(self) -> None:
        super().__init__(inplace=False)
