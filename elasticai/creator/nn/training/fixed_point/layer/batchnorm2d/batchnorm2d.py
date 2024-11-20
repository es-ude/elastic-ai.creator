from typing import Any

import torch

from elasticai.creator.base_modules.batchnorm2d import BatchNorm2d as BatchNorm2dBase
from elasticai.creator.nn.training.fixed_point import FixedPointConfigV2
from elasticai.creator.nn.training.fixed_point._math_operations import MathOperations


class BatchNorm2dTrainable(BatchNorm2dBase):
    def __init__(
        self,
        num_features: int,
        param_fxp_conf: FixedPointConfigV2,
        forward_fxp_conf: FixedPointConfigV2,
        backward_fxp_conf: FixedPointConfigV2,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Any = None,
    ) -> None:
        self.param_fxp_conf = param_fxp_conf
        self._forward_fxp_conf = forward_fxp_conf
        self._backward_fxp_conf = backward_fxp_conf
        super().__init__(
            operations=MathOperations(
                forward_config=forward_fxp_conf, backward_config=backward_fxp_conf
            ),
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
        )
        self.weight = torch.nn.Parameter(self.param_fxp_conf.quantize(self.weight))
        if self.bias:
            self.bias = torch.nn.Parameter(self.param_config.quantize(self.bias))
