from typing import Any, Callable, Protocol

from torch import Tensor
from torch.nn import BatchNorm2d as __BatchNorm2d

from elasticai.creator.base_modules.math_operations import Quantize
from elasticai.creator.nn.quantized_grads.quantized_parameters import (
    QuantizationSchemeByName,
    QuantizedParameters,
)


class _BatchNorm2d(__BatchNorm2d):
    """
    This class is a facade for batchnorm to allow clean multiple inheritance for BatchNorm2d.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Any = None,
        dtype: Any = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )


class MathOperations(Quantize, Protocol): ...


class BatchNorm2d(_BatchNorm2d, QuantizedParameters):
    """This module implements a 2d batch norm.
    The output of the batchnorm is fake quantized. The weights and bias are fake quantized during initialization.
    To keep the weights quantized use only optimizers that apply a quantized update.
    IMPORTANT: use only in-place operators for your quantize implementations
    """

    def __init__(
        self,
        operations: MathOperations,
        weight_quantization: Callable[[Tensor], None],
        bias_quantization: Callable[[Tensor], None],
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        params = [
            QuantizationSchemeByName("weight", weight_quantization),
            QuantizationSchemeByName("bias", bias_quantization),
        ]
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )
        self.qparams = params
        self._operations = operations
        weight_quantization(self.weight.data)
        bias_quantization(self.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        x = super().forward(x)
        return self._operations.quantize(x)
