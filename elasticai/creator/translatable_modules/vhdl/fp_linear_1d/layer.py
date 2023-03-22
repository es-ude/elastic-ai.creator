from typing import cast

import torch

from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.nn.linear import FixedPointLinear

from .design import FPLinear1d as FPLinearDesign


class FPLinear1d(FixedPointLinear):
    def translate(self) -> Design:
        def float_to_signed_int(value: float | list) -> int | list:
            if isinstance(value, list):
                return list(map(float_to_signed_int, value))
            return self.fixed_point_factory.as_integer(value)

        bias = torch.zeros(self.out_features) if self.bias is None else self.bias
        signed_int_weights = cast(
            list[list[int]], float_to_signed_int(self.weight.tolist())
        )
        signed_int_bias = cast(list[int], float_to_signed_int(bias.tolist()))

        return FPLinearDesign(
            frac_bits=self.frac_bits,
            total_bits=self.total_bits,
            in_feature_num=self.in_features,
            out_feature_num=self.out_features,
            weights=signed_int_weights,
            bias=signed_int_bias,
        )
