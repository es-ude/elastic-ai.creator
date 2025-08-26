from typing import Any, cast

import torch

from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.creator.base_modules.conv1d import Conv1d as Conv1dBase
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point.conv1d.design import Conv1dDesign
from elasticai.creator.nn.fixed_point.math_operations import MathOperations


class BatchNormedConv1d(DesignCreatorModule):
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
        self._config = FxpArithmetic(
            FxpParams(total_bits=total_bits, frac_bits=frac_bits)
        )
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

    @property
    def conv_weight(self) -> torch.Tensor:
        return self._conv1d.weight

    @property
    def conv_bias(self) -> torch.Tensor:
        return self._conv1d.bias

    @property
    def bn_weight(self) -> torch.Tensor:
        return self._batch_norm.weight

    @property
    def bn_bias(self) -> torch.Tensor:
        return self._batch_norm.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        has_batches = x.dim() == 3

        if not has_batches:
            x = x.view(1, *x.shape)

        x = self._conv1d(x)
        x = self._batch_norm(x)
        x = self._operations.quantize(x)

        if not has_batches:
            x = x.squeeze(dim=0)

        return x

    def create_design(self, name: str) -> Conv1dDesign:
        def float_to_signed_int(value: float | list) -> int | list:
            if isinstance(value, list):
                return list(map(float_to_signed_int, value))
            return self._operations.config.cut_as_integer(value)

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
        )
