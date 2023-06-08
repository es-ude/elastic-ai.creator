import torch


class ReLU(torch.nn.ReLU):
    def __init__(self) -> None:
        super().__init__()
