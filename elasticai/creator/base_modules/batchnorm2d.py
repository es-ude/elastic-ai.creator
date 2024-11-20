from typing import Any, Protocol

from torch import Tensor
from torch.nn import BatchNorm2d as _BatchNorm2d
from torch.nn.functional import batch_norm

from elasticai.creator.base_modules.math_operations import Quantize


class MathOperations(Quantize, Protocol):
    ...


class BatchNorm2d(_BatchNorm2d):
    def __init__(
        self,
        operations: MathOperations,
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
        self._operations = operations

    def forward(self, x: Tensor) -> Tensor:
        convolved = batch_norm(
            input=x,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
        )
        return self._operations.quantize(convolved)
