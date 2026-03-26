from torch import Tensor, nn
from torch.nn.utils.parametrize import register_parametrization

from elasticai.creator.nn.binary import MathOperations


class _Bin(nn.Module):
    def __init__(self):
        super().__init__()
        self._ops = MathOperations()

    def forward(self, input: Tensor) -> Tensor:
        return self._ops.quantize(input)


def binarize():
    return _Bin()


_ops = MathOperations()


class _Conv1d(nn.Conv1d):
    def forward(self, input: Tensor) -> Tensor:
        return super().forward(_ops.quantize(input))

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, binarized=True"


def conv1d(kernel_size: int, in_channels: int, out_channels: int) -> nn.Conv1d:
    c = _Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        bias=False,
        padding="valid",
    )
    register_parametrization(c, "weight", _Bin())
    return c
