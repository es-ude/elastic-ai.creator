from typing import Callable

import torch

from elasticai.creator.vhdl.number_representations import (
    FixedPointFactory,
    fixed_point_params_from_factory,
)
from elasticai.creator.vhdl.quantized_modules._autograd_functions.fixed_point_quantization import (
    FixedPointDequantFunction,
    FixedPointQuantFunction,
)
from elasticai.creator.vhdl.quantized_modules.quantized_module import QuantType

OperationType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class _LinearBase(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        matmul_op: OperationType = torch.matmul,
        add_op: OperationType = torch.add,
        input_quant: QuantType = lambda x: x,
        input_dequant: QuantType = lambda x: x,
        param_quant: QuantType = lambda x: x,
        param_dequant: QuantType = lambda x: x,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.matmul_op = matmul_op
        self.add_op = add_op
        self.input_quant = input_quant
        self.input_dequant = input_dequant
        self.param_quant = param_quant
        self.param_dequant = param_dequant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dq_input = self.input_dequant(self.input_quant(x))
        dq_weight = self.param_dequant(self.param_quant(self.weight))

        if self.bias is not None:
            dq_bias = self.param_dequant(self.param_quant(self.bias))
            return self.add_op(self.matmul_op(dq_input, dq_weight.T), dq_bias)

        return self.matmul_op(dq_input, dq_weight.T)

    def quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("The quantized_forward function is not implemented.")


class FixedPointLinear(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        fixed_point_factory: FixedPointFactory,
        bias: bool = True,
        device=None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            input_quant=lambda x: FixedPointQuantFunction.apply(x, fixed_point_factory),
            input_dequant=lambda x: FixedPointDequantFunction.apply(
                x, fixed_point_factory
            ),
            param_quant=lambda x: FixedPointQuantFunction.apply(x, fixed_point_factory),
            param_dequant=lambda x: FixedPointDequantFunction.apply(
                x, fixed_point_factory
            ),
            device=device,
        )
        self.fixed_point_factory = fixed_point_factory

    def quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        def fp_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            _, frac_bits = fixed_point_params_from_factory(self.fixed_point_factory)
            return (torch.matmul(a, b) / (1 << frac_bits)).floor()

        q_weight = self.param_quant(self.weight)

        if self.bias is not None:
            q_bias = self.param_quant(self.bias)
            return self.add_op(fp_matmul(x, q_weight.T), q_bias)

        return fp_matmul(x, q_weight.T)
