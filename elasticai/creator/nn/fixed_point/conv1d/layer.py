from typing import Any, cast

import torch

from elasticai.creator.base_modules.conv1d import Conv1d as Conv1dBase
from elasticai.creator.nn.fixed_point._math_operations import MathOperations
from elasticai.creator.nn.fixed_point._two_complement_fixed_point_config import (
    FixedPointConfig,
)
from elasticai.creator.vhdl.translatable import Translatable

from .design import Conv1d as Conv1dDesign


class Conv1d(Translatable, Conv1dBase):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        in_channels: int,
        out_channels: int,
        signal_length: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
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
            stride=stride,
            padding=padding,
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

    def translate(self, name: str) -> Conv1dDesign:
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
            stride=flatten_tuple(self.stride),
            padding=flatten_tuple(cast(int | tuple[int], self.padding)),
            dilation=flatten_tuple(self.dilation),
        )


class BatchNormedConv1d(Translatable, torch.nn.Module):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        in_channels: int,
        out_channels: int,
        signal_length: int,
        kernel_size: int | tuple[int],
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        bn_affine: bool = True,
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        bias: bool = True,
        device: Any = None,
    ) -> None:
        super().__init__()
        self._config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
        self._operations = MathOperations(config=self._config)
        self._signal_length = signal_length
        self._conv1d = Conv1dBase(
            operations=self._operations,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            device=device,
        )
        self._batch_norm = torch.nn.BatchNorm1d(
            num_features=out_channels,
            eps=bn_eps,
            momentum=bn_momentum,
            affine=bn_affine,
            track_running_stats=True,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        has_batches = x.dim() == 2
        input_shape = (
            (x.shape[0], self._conv1d.in_channels, -1)
            if has_batches
            else (1, self._conv1d.in_channels, -1)
        )
        output_shape = (x.shape[0], -1) if has_batches else (-1,)

        x = x.view(*input_shape)
        x = self._conv1d(x)
        x = self._batch_norm(x)
        x = self._operations.quantize(x)

        return x.view(*output_shape)

    def translate(self, name: str) -> Conv1dDesign:
        def float_to_signed_int(value: float | list) -> int | list:
            if isinstance(value, list):
                return list(map(float_to_signed_int, value))
            return self._operations.config.as_integer(value)

        def flatten_tuple(x: int | tuple[int, ...]) -> int:
            return x[0] if isinstance(x, tuple) else x

        bn_mean = cast(torch.Tensor, self._batch_norm.running_mean)
        bn_variance = cast(torch.Tensor, self._batch_norm.running_var)
        bn_epsilon = self._batch_norm.eps
        conv_weight = self._conv1d.weight
        conv_bias = (
            torch.tensor([0] * self._conv1d.out_channels)
            if self._conv1d.bias is None
            else self._conv1d.bias
        )

        std = torch.sqrt(bn_variance + bn_epsilon)
        weights = conv_weight / std
        bias = (conv_bias - bn_mean) / std

        if self._batch_norm.affine:
            weights = (self._batch_norm.weight * weights.t()).t()
            bias = self._batch_norm.weight * bias + self._batch_norm.bias

        return Conv1dDesign(
            name=name,
            total_bits=self._config.total_bits,
            frac_bits=self._config.frac_bits,
            in_channels=self._conv1d.in_channels,
            out_channels=self._conv1d.out_channels,
            signal_length=self._signal_length,
            kernel_size=flatten_tuple(self._conv1d.kernel_size),
            weights=cast(list[list[list[int]]], float_to_signed_int(weights.tolist())),
            bias=cast(list[int], float_to_signed_int(bias.tolist())),
            stride=flatten_tuple(self._conv1d.stride),
            padding=flatten_tuple(cast(int | tuple[int], self._conv1d.padding)),
            dilation=flatten_tuple(self._conv1d.dilation),
        )
