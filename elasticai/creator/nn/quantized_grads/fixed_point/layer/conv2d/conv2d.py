import warnings
from typing import Any, cast

import torch

from elasticai.creator.base_modules.conv2d import Conv2d as Conv2dBase
from elasticai.creator.nn.quantized_grads.fixed_point._math_operations import (
    MathOperations,
)
from elasticai.creator.nn.quantized_grads.fixed_point._two_complement_fixed_point_config import (
    FixedPointConfigV2,
)


class Conv2dTrainable(Conv2dBase):
    def __init__(
        self,
        param_fxp_conf: FixedPointConfigV2,
        forward_fxp_config: FixedPointConfigV2,
        backward_fxp_config: FixedPointConfigV2,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        padding: int | tuple[int, int] | str = 0,
        bias: bool = True,
        device: Any = None,
    ) -> None:
        self.param_config = param_fxp_conf
        self._forward_config = forward_fxp_config
        self._backward_config = backward_fxp_config
        super().__init__(
            operations=MathOperations(
                forward_config=self._forward_config,
                backward_config=self._backward_config,
            ),
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            device=device,
        )
        self.weight = torch.nn.Parameter(self.param_config.quantize(self.weight))
        if bias:
            self.bias = torch.nn.Parameter(self.param_config.quantize(self.bias))
