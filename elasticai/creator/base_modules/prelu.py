import torch


class PReLU(torch.nn.PReLU):
    def __init__(self, init: float = 0.25) -> None:
        super().__init__(num_parameters=1, init=init)
