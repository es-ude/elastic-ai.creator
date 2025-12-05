from typing import Union

from torch import nn, Tensor
from torch.nn import Conv1d as TorchConv1d
from torch.nn.utils import parametrize as P
from torch.nn.common_types import _size_1_t

from elasticai.creator_plugins.quantized_grads.linear_quantization.base_modules.linear_quantization_module import \
    LinearQuantizationLayer
from elasticai.creator_plugins.quantized_grads.linear_quantization.module_quantization import QuantizationModule
from elasticai.creator_plugins.quantized_grads.linear_quantization.param_quantization import \
    ParamQuantizationSimulatedModule, ParamQuantizationModule


class Conv1d(LinearQuantizationLayer, TorchConv1d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_1_t,
            input_quantization: QuantizationModule,
            output_quantization: QuantizationModule,
            weight_quantization: ParamQuantizationModule,
            bias_quantization: ParamQuantizationSimulatedModule,
            stride: _size_1_t = 1,
            padding: Union[str, _size_1_t] = 0,
            dilation: _size_1_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",  # TODO: refine this type
            device=None,
            dtype=None,
    ) -> None:
        if bias ^ isinstance(bias_quantization, ParamQuantizationSimulatedModule):
            raise Exception(
                f"if bias is True, bias_quantization can needs be set. "
                f"If not it is not allowed to be set."
                f"You have choosen {bias=} and {bias_quantization=}."
            )
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
        P.register_parametrization(self, "weight", weight_quantization)
        if bias:
            P.register_parametrization(self, "bias", bias_quantization)
        self.add_module("input_quantization", input_quantization)
        self.add_module("output_quantization", output_quantization)

    def forward(self, x: Tensor) -> Tensor:
        x_new, x_scale, x_zero_point = self.input_quantization.quantize(x)
        w, w_scale, w_zero_point = self.parametrizations["weight"][0].quantize(self.weight)
        y_int = self._conv_forward(x_new-x_zero_point, w-w_zero_point, self.bias)
        y = self.output_quantization.quantize_simulated(y_int) * w_scale * x_scale
        return y