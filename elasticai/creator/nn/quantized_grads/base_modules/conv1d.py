from typing import Any, Callable, Protocol

from torch import Tensor
from torch.nn import Conv1d as _Conv1d
from torch.nn.functional import conv1d

from elasticai.creator.base_modules.math_operations import Quantize


class MathOperations(Quantize, Protocol): ...


class Conv1d(_Conv1d):
    """This module implements a 1d convolution.
    The output of the convolution is fake quantized. The weights and bias are fake quantized during initialization.
    To keep the weights quantized use only optimizers that apply a quantized update
    """

    def __init__(
        self,
        operations: MathOperations,
        param_quantization: Callable[[Tensor], Tensor],
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
        self.param_quantization = param_quantization
        self.weight.data = param_quantization(self.weight.data)
        self.bias.data = param_quantization(self.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        convolved = conv1d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return self._operations.quantize(convolved)
