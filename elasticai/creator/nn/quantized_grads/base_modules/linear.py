from typing import Any, Callable, Protocol

import torch

from elasticai.creator.base_modules.math_operations import Add, MatMul, Quantize


class MathOperations(Quantize, Add, MatMul, Protocol): ...


class Linear(torch.nn.Linear):
    """This module implements a linear layer.
    The output of the convolution is fake quantized. The weights and bias are fake quantized during initialization.
    To keep the weights quantized use only optimizers that apply a quantized update
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        operations: MathOperations,
        param_quantization: Callable[[torch.Tensor], torch.Tensor],
        bias: bool,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self._operations = operations
        self.param_quantization = param_quantization
        self.weight.data = param_quantization(self.weight.data)
        self.bias.data = param_quantization(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.bias is not None:
            return self._operations.add(
                self._operations.matmul(x, self.weight.T), self.bias
            )

        return self._operations.matmul(x, self.weight.T)
