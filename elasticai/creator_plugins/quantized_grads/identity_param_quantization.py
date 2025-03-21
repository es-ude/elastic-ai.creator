import torch.nn
from torch import Tensor


class Identity(torch.nn.Identity):
    @staticmethod
    def right_inverse(x: Tensor) -> Tensor:
        return x
