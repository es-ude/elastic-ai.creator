from typing import Any

from torch import Tensor
from torch.nn import Conv2d as TorchConv2d
from torch.nn import Module
from torch.nn.utils import parametrize as P


class Conv2d(TorchConv2d):
    """A 2d convolution.
    The weights and bias are fake quantized during initialization.
    Make sure that math_ops is a module where all needed tensors are part of it,
    so they can be moved to the same device.
    Make sure that weight_quantization and bias_quantization are modules that implement the forward function.
    If you want to quantize during initialization or only apply quantized updates make sure to use a quantized optimizer
    and implement the right_inverse method for your module.
    """

    def __init__(
        self,
        math_ops: Module,
        weight_quantization: Module,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] | str = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
        bias_quantization: Module = None,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        if bias ^ isinstance(bias_quantization, Module):
            raise Exception(
                f"if bias is True, bias_quantization can needs be set. "
                f"If not it is not allowed to be set."
                f"You have choosen {bias=} and {bias_quantization=}."
            )

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
        P.register_parametrization(self, "weight", weight_quantization)
        if bias:
            P.register_parametrization(self, "bias", bias_quantization)
        self.add_module("math_ops", math_ops)

    def forward(self, x: Tensor) -> Tensor:
        return self.math_ops(super().forward(x))
