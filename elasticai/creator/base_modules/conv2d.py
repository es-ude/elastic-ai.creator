from typing import Any, Protocol

from torch import Tensor
from torch.nn import Conv2d as _Conv2d
from torch.nn.functional import conv2d

from elasticai.creator.base_modules.math_operations import Quantize


class MathOperations(Quantize, Protocol):
    ...


class Conv2d(_Conv2d):
    def __init__(
        self,
        operations: MathOperations,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            device=device,
            dtype=dtype,
        )
        self._operations = operations

    def forward(self, x: Tensor) -> Tensor:
        convolved = conv2d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return self._operations.quantize(convolved)
