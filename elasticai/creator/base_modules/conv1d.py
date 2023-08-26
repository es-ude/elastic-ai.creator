from typing import Any

from torch import Tensor
from torch.nn import Conv1d as _Conv1d
from torch.nn.functional import conv1d

from elasticai.creator.base_modules.math_operations import Quantize as MathOperations


class Conv1d(_Conv1d):
    def __init__(
        self,
        operations: MathOperations,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] | str = 0,
        dilation: int | tuple[int] = 1,
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
        quantized_weights = self._operations.quantize(self.weight)
        quantized_bias = (
            self._operations.quantize(self.bias) if self.bias is not None else None
        )
        convolved = conv1d(
            input=x,
            weight=quantized_weights,
            bias=quantized_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return self._operations.quantize(convolved)
