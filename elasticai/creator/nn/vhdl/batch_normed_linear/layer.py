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
        total_bits: int,
        frac_bits: int,
        bias: bool,
        bn_eps: float,
        bn_momentum: float,
        bn_affine: bool,
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
        return FPLinear(
            in_feature_num=self._linear.in_features,
            out_feature_num=self._linear.out_features,
            total_bits=self._arithmetics.config.total_bits,
            frac_bits=self._arithmetics.config.frac_bits,
            weights=...,
            bias=...,
            name=name,
        )
