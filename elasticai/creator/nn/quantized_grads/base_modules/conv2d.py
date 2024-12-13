from typing import Any, Callable, Protocol

from torch import Tensor
from torch.nn import Conv2d as _Conv2d
from torch.nn.functional import conv2d

from elasticai.creator.base_modules.math_operations import Quantize
from elasticai.creator.nn.quantized_grads.quantized_parameters import (
    QuantizationSchemeByName,
    QuantizedParameters,
)


class MathOperations(Quantize, Protocol): ...


class Conv2d(_Conv2d, QuantizedParameters):
    """This module implements a 2d convolution.
    The output of the convolution is fake quantized. The weights and bias are fake quantized during initialization.
    To keep the weights quantized use only optimizers that apply a quantized update
    """

    def __init__(
        self,
        operations: MathOperations,
        weight_quantization: Callable[[Tensor], None],
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] | str = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
        bias_quantization: Callable[[Tensor], None] = None,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        _Conv2d.__init__(
            self,
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
        if bias ^ isinstance(bias_quantization, Callable):
            raise Exception(
                f"if bias is True, bias_quantization can needs be set. "
                f"If not it is not allowed to be set."
                f"You have choosen {bias=} and {bias_quantization=}."
            )
        self._operations = operations
        weight_quantization(self.weight.data)

        params = [QuantizationSchemeByName("weight", weight_quantization)]
        if bias is not False:
            bias_quantization(self.bias.data)
            params.append(QuantizationSchemeByName("bias", bias_quantization))
        QuantizedParameters.__init__(self, params)

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
