import torch


class Identity(torch.nn.Identity):
    def __init__(self) -> None:
        super().__init__()
