from typing import Any

import torch

from elasticai.creator.nn.two_complement_fixed_point_config import FixedPointConfig


class FixedPointQuantFunction(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        raise NotImplementedError()

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) != 2:
            raise TypeError(
                "apply() takes exactly two arguments "
                "(x: torch.Tensor, fixed_point_factory: Callable[[float], FixedPoint])"
            )
        x: torch.Tensor = args[0]
        config: FixedPointConfig = args[1]

        fp_ints = config.as_integer(x)
        out_of_bounds = fp_ints[config.integer_out_of_bounds(fp_ints)]
        if torch.any(out_of_bounds):
            raise ValueError("Cannot quantize tensor. Values out of bounds.")

        return fp_ints

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return *grad_outputs, None


class FixedPointDequantFunction(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        raise NotImplementedError()

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) != 2:
            raise TypeError(
                "apply() takes exactly two arguments "
                "(x: torch.Tensor, fixed_point_factory: Callable[[float], FixedPoint])"
            )
        x: torch.Tensor = args[0]
        config: FixedPointConfig = args[1]
        fp_values = config.as_rational(x)
        out_of_bound_coefficients = fp_values[config.rational_out_of_bounds(fp_values)]
        if torch.any(out_of_bound_coefficients):
            raise ValueError("Cannot dequantize tensor. Values out of bounds.")

        return fp_values

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return *grad_outputs, None
