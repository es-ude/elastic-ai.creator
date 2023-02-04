import torch


class HardTanh(torch.nn.Hardtanh):
    def __init__(
        self, min_val: float = -1, max_val: float = 1, inplace: bool = False
    ) -> None:
        super().__init__(min_val=min_val, max_val=max_val, inplace=inplace)
