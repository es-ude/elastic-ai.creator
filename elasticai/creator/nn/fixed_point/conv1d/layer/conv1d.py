from typing import Any, cast

import torch

from elasticai.creator.base_modules.conv1d import Conv1d as Conv1dBase
from elasticai.creator.nn.fixed_point._math_operations import MathOperations
from elasticai.creator.nn.fixed_point._two_complement_fixed_point_config import (
    FixedPointConfig,
)
from elasticai.creator.nn.fixed_point.conv1d.design import Conv1d as Conv1dDesign
from elasticai.creator.vhdl.design_creator import DesignCreator


class Conv1d(DesignCreator, Conv1dBase):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        in_channels: int,
        out_channels: int,
        signal_length: int,
        kernel_size: int | tuple[int],
        bias: bool = True,
        device: Any = None,
    ) -> None:
        self._config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
        self._signal_length = signal_length
        super().__init__(
            operations=MathOperations(config=self._config),
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        has_batches = x.dim() == 2

        input_shape = (
            (x.shape[0], self.in_channels, -1)
            if has_batches
            else (self.in_channels, -1)
        )
        output_shape = (x.shape[0], -1) if has_batches else (-1,)

        x = x.view(*input_shape)
        outputs = super().forward(x)
        return outputs.view(*output_shape)

    def create_design(self, name: str) -> Conv1dDesign:
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

        return Conv1dDesign(
            name=name,
            total_bits=self._config.total_bits,
            frac_bits=self._config.frac_bits,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            signal_length=self._signal_length,
            kernel_size=flatten_tuple(self.kernel_size),
            weights=signed_int_weights,
            bias=signed_int_bias,
        )
