from typing import Any, Callable

import torch

from elasticai.creator.vhdl.number_representations import (
    FixedPointFactory,
    fixed_point_params_from_factory,
)

OperationType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
QuantType = Callable[[torch.Tensor], torch.Tensor]


def _default_matmul_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


def _default_add_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.add(a, b)


def _identity_quant(x: torch.Tensor) -> torch.Tensor:
    return x


class _LinearBase(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        matmul_op: OperationType = _default_matmul_op,
        add_op: OperationType = _default_add_op,
        input_quant: QuantType = _identity_quant,
        input_dequant: QuantType = _identity_quant,
        param_quant: QuantType = _identity_quant,
        param_dequant: QuantType = _identity_quant,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self._matmul_op = matmul_op
        self._add_op = add_op
        self.input_quant = input_quant
        self.input_dequant = input_dequant
        self.param_quant = param_quant
        self.param_dequant = param_dequant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dq_input = self.input_dequant(self.input_quant(x))
        dq_weight = self.param_dequant(self.param_quant(self.weight))

        if self.bias is not None:
            dq_bias = self.param_dequant(self.param_quant(self.bias))
            return self._add_op(self._matmul_op(dq_input, dq_weight.T), dq_bias)

        return self._matmul_op(dq_input, dq_weight.T)


class _FixedPointQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, fixed_point_factory: FixedPointFactory
    ) -> torch.Tensor:
        total_bits, frac_bits = fixed_point_params_from_factory(fixed_point_factory)
        largest_fp_int = 2 ** (total_bits - 1) - 1
        return (x * (1 << frac_bits)).floor().clamp(-largest_fp_int, largest_fp_int)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return *grad_outputs, None


class _FixedPointDequantFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, fixed_point_factory: FixedPointFactory
    ) -> torch.Tensor:
        total_bits, frac_bits = fixed_point_params_from_factory(fixed_point_factory)
        min_value = 2 ** (total_bits - frac_bits - 1) * (-1)
        max_value = (2 ** (total_bits - 1) - 1) / (1 << frac_bits)
        return (x / (1 << frac_bits)).clamp(min_value, max_value)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return *grad_outputs, None


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
            input_quant=lambda x: _FixedPointQuantFunction.apply(
                x, fixed_point_factory
            ),
            input_dequant=lambda x: _FixedPointDequantFunction.apply(
                x, fixed_point_factory
            ),
            param_quant=lambda x: _FixedPointQuantFunction.apply(
                x, fixed_point_factory
            ),
            param_dequant=lambda x: _FixedPointDequantFunction.apply(
                x, fixed_point_factory
            ),
            device=device,
        )
