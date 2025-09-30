from typing import Any

from torch import Tensor
from torch.nn import Linear as TorchLinear
from torch.nn import Module
from torch.nn.utils import parametrize as P

from elasticai.creator_plugins.quantized_grads.linear_quantization.param_quantization import \
    ParamLinearQuantizationModule
from elasticai.creator_plugins.quantized_grads.linear_quantization.quantize_linear import quantize_linear_hte, \
    dequantize_linear


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
            math_ops: Module,
            in_features: int,
            out_features: int,
            weight_quantization: ParamLinearQuantizationModule,
            bias: bool,
            bias_quantization: ParamLinearQuantizationModule = None,
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
        self.input_num_values = 256
        P.register_parametrization(self, "weight", weight_quantization)
        if bias:
            P.register_parametrization(self, "bias", bias_quantization)
        self.add_module("math_ops", math_ops)

    def forward(self, x: Tensor) -> Tensor:
        x_new, x_scale, x_zero_point = self.parametrizations["weight"][0].quantize(x, self.parametrizations["weight"][0].min_value, self.parametrizations["weight"][0].max_value)
        w, w_scale, w_zero_point = self.parametrizations["weight"][0].quantize(self.weight, self.parametrizations["weight"][0].min_value, self.parametrizations["weight"][0].max_value)
        y_int =  (w - w_zero_point) @ (x_new-x_zero_point)

        y = (x_scale * w_scale) * y_int
        if self.bias:
            b, b_scale, b_zero_point = self.parametrizations["bias"][0].quantize(self.bias, self.parametrization["bias"][0].min_value, self.parametrizations["bias"][0].max_value)
            y += b_scale*(b-b_zero_point)
        print(f"{x_new=}, {x_scale=}, {w_zero_point=}")
        print(f"{y_int=}")
        print(f"{y=}")
        print(f"{1/x_scale*(w-w_zero_point)@x=}")
        print(f"{self.weight@x=}")

        y = self.math_ops(y)
        return y
