from typing import Any, Callable, Protocol

import torch
from torch.nn import Linear as _Linear

from elasticai.creator.base_modules.math_operations import Add, MatMul, Quantize
from elasticai.creator.nn.quantized_grads.quantized_parameters import (
    QuantizationSchemeByName,
    QuantizedParameters,
)


class MathOperations(Quantize, Add, MatMul, Protocol): ...


class Linear(_Linear, QuantizedParameters):
    """This module implements a linear layer.
    The output of the convolution is fake quantized. The weights and bias are fake quantized during initialization.
    To keep the weights quantized use only optimizers that apply a quantized update
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        operations: MathOperations,
        weight_quantization: Callable[[torch.Tensor], None],
        bias: bool,
        bias_quantization: Callable[[torch.Tensor], None] = None,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        if bias ^ isinstance(bias_quantization, Callable):
            raise Exception(
                f"if bias is True, bias_quantization can needs be set. "
                f"If not it is not allowed to be set."
                f"You have choosen {bias=} and {bias_quantization=}."
            )

        _Linear.__init__(self, in_features, out_features, bias, device, dtype)
        self._operations = operations
        weight_quantization(self.weight.data)
        params = [QuantizationSchemeByName("weight", weight_quantization)]
        if bias is not False:
            bias_quantization(self.bias.data)
            params.append(QuantizationSchemeByName("bias", bias_quantization))
        QuantizedParameters.__init__(self, params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return self._operations.add(
                self._operations.matmul(x, self.weight.T), self.bias
            )

        return self._operations.matmul(x, self.weight.T)
