from typing import Any, Callable, Protocol

from torch import Tensor
from torch.nn import BatchNorm2d as _BatchNorm2d

from elasticai.creator.base_modules.math_operations import Quantize


class MathOperations(Quantize, Protocol): ...


class BatchNorm2d(_BatchNorm2d):
    """This module implements a 2d batch norm.
    The output of the batchnorm is fake quantized. The weights and bias are fake quantized during initialization.
    To keep the weights quantized use only optimizers that apply a quantized update
    """

    def __init__(
        self,
        operations: MathOperations,
        param_quantization: Callable[[Tensor], Tensor],
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__(
            self,
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )
        self._operations = operations
        self.param_quantization = param_quantization
        self.weight.data = param_quantization(self.weight.data)
        self.bias.data = param_quantization(self.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        x = super().forward(x)
        return self._operations.quantize(x)
