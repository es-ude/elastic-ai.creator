import torch
from torch import Tensor

class Round(torch.autograd.Function):
    """
    Round deterministically to nearest neighbour with STE.
    """

    @staticmethod
    def forward(ctx, x, *args, **kwargs):
        return torch.round(x)

    @staticmethod
    def backward(ctx, *grad_output):
        return grad_output


def round_tensor(x: Tensor) -> Tensor:
    return Round.apply(x)

def _noise(number: Tensor) -> Tensor:
    return torch.rand_like(number)-0.5

