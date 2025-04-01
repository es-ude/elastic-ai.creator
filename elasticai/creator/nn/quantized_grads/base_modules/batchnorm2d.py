from typing import Any

from torch import Tensor
from torch.nn import BatchNorm2d as TorchBatchNorm2d
from torch.nn import Module
from torch.nn.utils import parametrize as P


class BatchNorm2d(TorchBatchNorm2d):
    """A BatchNorm2d.
    The output of the batchnorm is fake quantized. The weights and bias are fake quantized during initialization.
    Make sure that math_ops is a module where all needed tensors are part of it,
    so they can be moved to the same device.
    Make sure that weight_quantization and bias_quantization are modules that implement the forward function.
    If you want to quantize during initialization or only apply quantized updates make sure to use a quantized optimizer
    and implement the right_inverse method for your module.
    """

    def __init__(
        self,
        math_ops: Module,
        weight_quantization: Module,
        bias_quantization: Module,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )
        P.register_parametrization(self, "weight", weight_quantization)
        P.register_parametrization(self, "bias", bias_quantization)
        self.add_module("math_ops", math_ops)

    def forward(self, x: Tensor) -> Tensor:
        return self.math_ops(super().forward(x))
