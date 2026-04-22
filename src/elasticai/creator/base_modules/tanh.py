import torch


class Tanh(torch.nn.Tanh):
    def __init__(self) -> None:
        super().__init__()
