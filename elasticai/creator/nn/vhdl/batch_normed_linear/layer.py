from typing import Any, cast

import torch

from elasticai.creator.base_modules.autograd_functions.fixed_point_quantization import (
    FixedPointDequantFunction,
)
from elasticai.creator.base_modules.linear import Linear
from elasticai.creator.hdl.translatable import Translatable
from elasticai.creator.nn.fixed_point_arithmetics import FixedPointArithmetics
from elasticai.creator.nn.two_complement_fixed_point_config import FixedPointConfig
from elasticai.creator.nn.vhdl.linear.design import FPLinear


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
            device=device,
        )

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        return self._arithmetics.quantize(x)

    def _dequantize(self, x: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor, FixedPointDequantFunction.apply(x, self._arithmetics.config)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._linear(inputs)
        x = self._dequantize(x)
        x = self._batch_norm(x)
        x = self._quantize(x)
        return x

    def translate(self, name: str) -> FPLinear:
        def float_to_signed_int(value: float | list) -> int | list:
            if isinstance(value, list):
                return list(map(float_to_signed_int, value))
            return self._arithmetics.config.as_integer(value)

        bn_weight = self._batch_norm.weight
        bn_bias = self._batch_norm.bias
        bn_mean = cast(torch.Tensor, self._batch_norm.running_mean)
        bn_variance = cast(torch.Tensor, self._batch_norm.running_var)
        bn_epsilon = self._batch_norm.eps

        factor = bn_weight / torch.sqrt(bn_variance + bn_epsilon)

        new_weights = (factor * self._linear.weight.t()).t()
        new_bias = factor * (self._linear.bias - bn_mean) + bn_bias

        return FPLinear(
            in_feature_num=self._linear.in_features,
            out_feature_num=self._linear.out_features,
            total_bits=self._arithmetics.config.total_bits,
            frac_bits=self._arithmetics.config.frac_bits,
            weights=cast(list[list[int]], float_to_signed_int(new_weights.tolist())),
            bias=cast(list[int], float_to_signed_int(new_bias.tolist())),
            name=name,
        )
