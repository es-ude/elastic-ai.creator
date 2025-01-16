from typing import Any, cast

import torch

from elasticai.creator.base_modules.linear import Linear as LinearBase
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point.linear.design import LinearDesign
from elasticai.creator.nn.fixed_point.math_operations import MathOperations
from elasticai.creator.nn.fixed_point.two_complement_fixed_point_config import (
    FixedPointConfig,
)


class BatchNormedLinear(DesignCreatorModule, torch.nn.Module):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        bn_affine: bool = True,
        device: Any = None,
    ) -> None:
        super().__init__()
        self._operations = MathOperations(
            config=FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
        )
        self._linear = LinearBase(
            in_features=in_features,
            out_features=out_features,
            operations=self._operations,
            bias=bias,
            device=device,
        )
        self._batch_norm = torch.nn.BatchNorm1d(
            num_features=out_features,
            eps=bn_eps,
            momentum=bn_momentum,
            affine=bn_affine,
            track_running_stats=True,
            device=device,
        )

    @property
    def lin_weight(self) -> torch.Tensor:
        return self._linear.weight

    @property
    def lin_bias(self) -> torch.Tensor | None:
        return self._linear.bias

    @property
    def bn_weight(self) -> torch.Tensor:
        return self._batch_norm.weight

    @property
    def bn_bias(self) -> torch.Tensor:
        return self._batch_norm.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        has_batches = x.dim() == 2
        input_shape = x.shape if has_batches else (1, -1)
        output_shape = (x.shape[0], -1) if has_batches else (-1,)

        x = x.view(*input_shape)
        x = self._linear(x)
        x = self._batch_norm(x)
        x = self._operations.quantize(x)

        return x.view(*output_shape)

    def create_design(self, name: str) -> LinearDesign:
        def float_to_signed_int(value: float | list) -> int | list:
            if isinstance(value, list):
                return list(map(float_to_signed_int, value))
            return self._operations.config.as_integer(value)

        bn_mean = cast(torch.Tensor, self._batch_norm.running_mean)
        bn_variance = cast(torch.Tensor, self._batch_norm.running_var)
        bn_epsilon = self._batch_norm.eps
        lin_weight = self._linear.weight
        lin_bias = (
            torch.tensor([0] * self._linear.out_features)
            if self._linear.bias is None
            else self._linear.bias
        )

        std = torch.sqrt(bn_variance + bn_epsilon)
        weights = lin_weight / std
        bias = (lin_bias - bn_mean) / std

        if self._batch_norm.affine:
            weights = (self._batch_norm.weight * weights.t()).t()
            bias = self._batch_norm.weight * bias + self._batch_norm.bias

        return LinearDesign(
            in_feature_num=self._linear.in_features,
            out_feature_num=self._linear.out_features,
            total_bits=self._operations.config.total_bits,
            frac_bits=self._operations.config.frac_bits,
            weights=cast(list[list[int]], float_to_signed_int(weights.tolist())),
            bias=cast(list[int], float_to_signed_int(bias.tolist())),
            name=name,
        )
