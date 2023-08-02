from typing import Any, cast

import torch

from elasticai.creator.base_modules.arithmetics.fixed_point_arithmetics import (
    FixedPointArithmetics,
)
from elasticai.creator.base_modules.conv1d import Conv1d
from elasticai.creator.base_modules.two_complement_fixed_point_config import (
    FixedPointConfig,
)
from elasticai.creator.vhdl.design_base.design import Design
from elasticai.creator.vhdl.translatable import Translatable

from .design import FPConv1d as FPConv1dDesign


class FPConv1d(Translatable, Conv1d):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        in_channels: int,
        out_channels: int,
        signal_length: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] | str = 0,
        bias: bool = True,
        device: Any = None,
    ) -> None:
        self._config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
        self._signal_length = signal_length
        super().__init__(
            arithmetics=FixedPointArithmetics(config=self._config),
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            device=device,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        has_batches = inputs.dim() == 2

        input_shape = (
            (inputs.shape[0], self.in_channels, -1)
            if has_batches
            else (self.in_channels, -1)
        )
        output_shape = (inputs.shape[0], -1) if has_batches else (-1,)

        inputs = inputs.view(*input_shape)
        outputs = super().forward(inputs)
        return outputs.view(*output_shape)

    def translate(self, name: str) -> FPConv1dDesign:
        def float_to_signed_int(value: float | list) -> int | list:
            if isinstance(value, list):
                return list(map(float_to_signed_int, value))
            return self._config.as_integer(value)

        def flatten_tuple(x: int | tuple[int, ...]) -> int:
            return x[0] if isinstance(x, tuple) else x

        bias = [0] * self.out_channels if self.bias is None else self.bias.tolist()
        signed_int_weights = cast(
            list[list[list[int]]], float_to_signed_int(self.weight.tolist())
        )
        signed_int_bias = cast(list[int], float_to_signed_int(bias))

        return FPConv1dDesign(
            name=name,
            total_bits=self._config.total_bits,
            frac_bits=self._config.frac_bits,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            signal_length=self._signal_length,
            kernel_size=flatten_tuple(self.kernel_size),
            weights=signed_int_weights,
            bias=signed_int_bias,
            stride=flatten_tuple(self.stride),
            padding=flatten_tuple(self.padding),
            dilation=flatten_tuple(self.dilation),
        )
