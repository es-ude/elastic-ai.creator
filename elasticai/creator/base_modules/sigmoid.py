import torch


class Sigmoid(torch.nn.Sigmoid):
    def __init__(self) -> None:
        super().__init__()
