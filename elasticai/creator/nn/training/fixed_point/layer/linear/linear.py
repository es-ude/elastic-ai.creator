from typing import Any

import torch

from elasticai.creator.base_modules.linear import Linear as LinearBase
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.training.fixed_point._math_operations import MathOperations
from elasticai.creator.nn.training.fixed_point._two_complement_fixed_point_config import (
    FixedPointConfigV2,
)
from elasticai.creator.vhdl.design.design import Design


class LinearTrainable(LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        param_fxp_conf: FixedPointConfigV2,
        forward_fxp_conf: FixedPointConfigV2,
        grad_fxp_conf: FixedPointConfigV2,
        bias: bool = True,
        device: Any = None,
    ) -> None:
        self.param_config = param_fxp_conf
        self._forward_config = forward_fxp_conf
        self._grad_config = grad_fxp_conf
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            operations=MathOperations(
                config=self._forward_config, grad_config=self._grad_config
            ),
            bias=bias,
            device=device,
        )
        self.weight = torch.nn.Parameter(self.param_config.round(self.weight))
        if bias:
            self.bias = torch.nn.Parameter(self.param_config.round(self.bias))
