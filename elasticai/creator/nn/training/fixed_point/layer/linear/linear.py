from typing import Any

import torch

from elasticai.creator.base_modules.linear import Linear as LinearBase
from elasticai.creator.nn.training.fixed_point._math_operations import MathOperations
from elasticai.creator.nn.training.fixed_point._two_complement_fixed_point_config import (
    FixedPointConfigV2,
)


class LinearTrainable(LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        param_fxp_conf: FixedPointConfigV2,
        forward_fxp_conf: FixedPointConfigV2,
        backard_fxp_conf: FixedPointConfigV2,
        bias: bool = True,
        device: Any = None,
    ) -> None:
        self.param_config = param_fxp_conf
        self._forward_config = forward_fxp_conf
        self._backward_config = backard_fxp_conf
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            operations=MathOperations(
                forward_config=self._forward_config,
                backward_config=self._backward_config,
            ),
            bias=bias,
            device=device,
        )
        self.weight = torch.nn.Parameter(self.param_config.quantize(self.weight))
        if bias:
            self.bias = torch.nn.Parameter(self.param_config.quantize(self.bias))
