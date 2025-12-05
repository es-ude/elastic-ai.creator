from typing import Any

import torch.nn.functional
from torch import Tensor
from torch.nn import Linear as TorchLinear
from torch.nn import Module
from torch.nn.utils import parametrize as P

from elasticai.creator_plugins.quantized_grads.linear_quantization.param_quantization import \
    ParamQuantizationModule


class Linear(TorchLinear):
    """A linear layer.
    The weights and bias are fake quantized during initialization.
    Make sure that math_ops is a module where all needed tensors are part of it,
    so they can be moved to the same device.
    Make sure that weight_quantization and bias_quantization are modules that implement the forward function.
    If you want to quantize during initialization or only apply quantized updates make sure to use a quantized optimizer
    and implement the right_inverse method for your module.
    """

    def __init__(
            self,
            input_quantization: Module,
            output_quantization: Module,
            in_features: int,
            out_features: int,
            weight_quantization: ParamQuantizationModule,
            bias: bool,
            bias_quantization: ParamQuantizationModule = None,
            device: Any = None,
            dtype: Any = None,
    ) -> None:
        if bias ^ isinstance(bias_quantization, Module):
            raise Exception(
                f"if bias is True, bias_quantization can needs be set. "
                f"If not it is not allowed to be set."
                f"You have choosen {bias=} and {bias_quantization=}."
            )

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        P.register_parametrization(self, "weight", weight_quantization)
        if bias:
            P.register_parametrization(self, "bias", bias_quantization)
        self.add_module("output_quantization", output_quantization)
        self.add_module("input_quantization", input_quantization)

    def forward(self, x: Tensor) -> Tensor:
        x_new, x_scale, x_zero_point = self.input_quantization(x)
        w, w_scale, w_zero_point = self.parametrizations["weight"][0].quantize(self.weight)
        y_int =  (x_new+x_zero_point) @ (w + w_zero_point).T
        if self.bias is not None:
            y_int = y_int + self.bias
        y = self.output_quantization(y_int) * w_scale* x_scale
        return y

