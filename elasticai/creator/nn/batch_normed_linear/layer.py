from typing import Any, cast

import torch

from elasticai.creator.base_modules.arithmetics.fixed_point_arithmetics import (
    FixedPointArithmetics,
)
from elasticai.creator.base_modules.linear import Linear
from elasticai.creator.base_modules.two_complement_fixed_point_config import (
    FixedPointConfig,
)
from elasticai.creator.nn.linear.design import FPLinear
from elasticai.creator.vhdl.translatable import Translatable


class FPBatchNormedLinear(Translatable, torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        total_bits: int,
        frac_bits: int,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        bn_affine: bool = True,
        device: Any = None,
    ) -> None:
        super().__init__()
        self._arithmetics = FixedPointArithmetics(
            config=FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
        )
        self._linear = Linear(
            in_features=in_features,
            out_features=out_features,
            arithmetics=self._arithmetics,
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._linear(inputs)
        x = self._batch_norm(x)
        return self._arithmetics.quantize(x)

    def translate(self, name: str) -> FPLinear:
        def float_to_signed_int(value: float | list) -> int | list:
            if isinstance(value, list):
                return list(map(float_to_signed_int, value))
            return self._arithmetics.config.as_integer(value)

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

        return FPLinear(
            in_feature_num=self._linear.in_features,
            out_feature_num=self._linear.out_features,
            total_bits=self._arithmetics.config.total_bits,
            frac_bits=self._arithmetics.config.frac_bits,
            weights=cast(list[list[int]], float_to_signed_int(weights.tolist())),
            bias=cast(list[int], float_to_signed_int(bias.tolist())),
            name=name,
        )
