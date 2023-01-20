from typing import Any

import torch

from elasticai.creator.vhdl.number_representations import FixedPointFactory


class FixedPointQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) != 2:
            raise TypeError(
                "apply() takes exactly two arguments "
                "(x: torch.Tensor, fixed_point_factory: Callable[[float], FixedPoint])"
            )
        x: torch.Tensor = args[0]
        fp_factory: FixedPointFactory = args[1]
        total_bits, frac_bits = fp_factory.total_bits, fp_factory.frac_bits

        fp_ints = (x * (1 << frac_bits)).int().float()

        min_fp_int = 2 ** (total_bits - 1) * (-1)
        max_fp_int = 2 ** (total_bits - 1) - 1
        out_of_bounds = fp_ints[(fp_ints < min_fp_int) | (fp_ints > max_fp_int)]

        if torch.any(out_of_bounds):
            raise ValueError("Cannot quantize tensor. Values out of bounds.")

        return fp_ints

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return *grad_outputs, None


class FixedPointDequantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) != 2:
            raise TypeError(
                "apply() takes exactly two arguments "
                "(x: torch.Tensor, fixed_point_factory: Callable[[float], FixedPoint])"
            )
        x: torch.Tensor = args[0]
        fp_factory: FixedPointFactory = args[1]
        total_bits, frac_bits = fp_factory.total_bits, fp_factory.frac_bits

        fp_values = x / (1 << frac_bits)

        min_fp = -1 * (1 << (total_bits - 1)) / (1 << frac_bits)
        max_fp = int("1" * (total_bits - 1), 2) / (1 << frac_bits)
        out_of_bounds = fp_values[(fp_values < min_fp) | (fp_values > max_fp)]

        if torch.any(out_of_bounds):
            raise ValueError("Cannot dequantize tensor. Values out of bounds.")

        return fp_values

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return *grad_outputs, None
