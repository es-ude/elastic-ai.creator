from typing import Any

import torch


class RoundToFloat(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) != 3:
            raise TypeError(
                "apply() takes exactly three arguments "
                "(x: torch.Tensor, mantissa_bits: int, exponent_bits: int)"
            )
        x: torch.Tensor = args[0]
        mantissa_bits: int = args[1]
        exponent_bits: int = args[2]

        exponent_bias = 2 ** (exponent_bits - 1)
        largest_value = (2 - 1 / 2**mantissa_bits) * 2 ** (
            2**exponent_bits - exponent_bias - 1
        )

        out_of_bounds = (x < -largest_value) | (x > largest_value)
        if torch.any(out_of_bounds):
            raise ValueError("Cannot quantize tensor. Values out of bounds.")

        smallest_value = 2 ** (1 - exponent_bias - mantissa_bits)
        x[(x > -smallest_value) & (x < smallest_value)] = smallest_value

        scale = 2 ** (x.abs().log2().floor() - mantissa_bits)
        return scale * torch.round(x / scale)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return *grad_outputs, None, None
