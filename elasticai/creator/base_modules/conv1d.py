from typing import Any

import torch

from .arithmetics.arithmetics import Arithmetics


class Conv1d(torch.nn.Conv1d):
    def __init__(
        self,
        arithmetics: Arithmetics,
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
        self._arithmetics = arithmetics

    @staticmethod
    def _flatten_tuple(x: int | tuple[int, ...]) -> int:
        return x[0] if isinstance(x, tuple) else x

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        quantized_weights = self._arithmetics.quantize(self.weight)
        quantized_bias = (
            self._arithmetics.quantize(self.bias) if self.bias is not None else None
        )
        return self._arithmetics.conv1d(
            inputs=inputs,
            weights=quantized_weights,
            bias=quantized_bias,
            stride=self._flatten_tuple(self.stride),
            padding=(
                self.padding
                if isinstance(self.padding, str)
                else self._flatten_tuple(self.padding)
            ),
            dilation=self._flatten_tuple(self.dilation),
            groups=self.groups,
        )
