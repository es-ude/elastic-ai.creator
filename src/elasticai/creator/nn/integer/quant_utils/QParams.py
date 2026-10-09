from typing import Protocol

import torch
import torch.nn as nn

from elasticai.creator.nn.integer._dtypes import FLOAT_DTYPES, INT_DTYPES, check_dtype
from elasticai.creator.nn.integer.math_operations.math_operations import MathOperations
from elasticai.creator.nn.integer.quant_utils.calculate_quant_params import (
    calculate_asymmetric_quant_params,
    calculate_symmetric_quant_params,
)
from elasticai.creator.nn.integer.quant_utils.Observers import BaseObserver


class QParams(nn.Module):
    """Quantization parameters for one tensor: scale factor, zero point, range.

    The float range is tracked by `observer`. The integer range follows from
    `quant_bits`, `is_signed` and `is_symmetric`; for 8 bits it is
    `[-128, 127]` (asymmetric signed), `[-127, 127]` (symmetric signed),
    `[0, 255]` (asymmetric unsigned) or `[0, 254]` (symmetric unsigned).

    Args:
        quant_bits: Bit width of the quantized values.
        observer: Observer that tracks the float range in its `min_float` and
            `max_float` buffers.
        is_symmetric: Use a symmetric range with zero point 0.
        is_signed: Use a signed integer range.
    """

    min_float: torch.Tensor
    max_float: torch.Tensor
    min_quant: torch.Tensor
    max_quant: torch.Tensor
    scale_factor: torch.Tensor
    zero_point: torch.Tensor
    eps: torch.Tensor

    def __init__(
        self,
        quant_bits: int,
        observer: BaseObserver,
        is_symmetric: bool,
        is_signed: bool,
    ):
        super().__init__()

        self.observer = observer
        self.quant_bits = quant_bits
        self.is_symmetric = is_symmetric
        self.is_signed = is_signed

        self.register_buffer(
            "min_float", torch.ones((1), dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer(
            "max_float", torch.ones((1), dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer(
            "min_quant", torch.zeros((1), dtype=torch.int32, requires_grad=False)
        )
        self.register_buffer(
            "max_quant", torch.zeros((1), dtype=torch.int32, requires_grad=False)
        )
        self.register_buffer(
            "scale_factor", torch.ones((1), dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer(
            "zero_point", torch.zeros((1), dtype=torch.int32, requires_grad=False)
        )
        self.register_buffer(
            "eps",
            torch.tensor(
                (torch.finfo(torch.float32).eps),
                dtype=torch.float32,
                requires_grad=False,
            ),
        )

        self._calculate_quant_range()
        self.math_ops = MathOperations()

    def _calculate_quant_range(self):
        if self.is_signed:
            min_value = -(2 ** (self.quant_bits - 1)) + (
                1 if self.is_symmetric else 0
            )  # -127 or -128 for 8 bits
            self.min_quant.copy_(torch.tensor(min_value))

            self.max_quant.copy_(
                torch.tensor((2 ** (self.quant_bits - 1)) - 1)  # 127 for 8 bits
            )
        else:
            self.min_quant.copy_(torch.tensor(0))  # 0 for 8 bits
            max_value = (2**self.quant_bits) - (
                2 if self.is_symmetric else 1
            )  # 254 or 255 for 8 bits
            self.max_quant.copy_(torch.tensor(max_value))

    def update_quant_params(self, x: torch.Tensor) -> None:
        """Observe `x` and recalculate scale factor, zero point and float range.

        Args:
            x: Float tensor to observe.

        Raises:
            TypeError: If `x` is not a `torch.Tensor` or does not have a
                floating-point dtype.
        """
        check_dtype("x", x, *FLOAT_DTYPES)
        self.observer(x)
        calculate_quant_params_fn = (
            calculate_symmetric_quant_params
            if self.is_symmetric
            else calculate_asymmetric_quant_params
        )
        scale_factor, zero_point, min_float, max_float = calculate_quant_params_fn(
            min_quant=self.min_quant,
            max_quant=self.max_quant,
            min_float=self.observer.min_float,
            max_float=self.observer.max_float,
            eps=self.eps,
        )

        for attr, value in zip(
            [self.scale_factor, self.zero_point, self.min_float, self.max_float],
            [scale_factor, zero_point, min_float, max_float],
        ):
            attr.copy_(value)

    def set_scale_factor(self, given_scale_factor: torch.Tensor) -> None:
        """Overwrite the scale factor.

        Args:
            given_scale_factor: The new scale factor.

        Raises:
            TypeError: If `given_scale_factor` is not a `torch.Tensor` or does not
                have a floating-point dtype.
        """
        check_dtype("given_scale_factor", given_scale_factor, *FLOAT_DTYPES)
        self.scale_factor.copy_(given_scale_factor)

    def set_zero_point(self, given_zero_point: torch.Tensor) -> None:
        """Overwrite the zero point.

        Args:
            given_zero_point: The new zero point.

        Raises:
            TypeError: If `given_zero_point` is not a `torch.Tensor` or does not
                have an integer dtype.
        """
        check_dtype("given_zero_point", given_zero_point, *INT_DTYPES)
        self.zero_point.copy_(given_zero_point)

    def set_quant_range(self, given_quant_bits: int) -> None:
        """Change the bit width and recalculate the integer range.

        Args:
            given_quant_bits: The new bit width.
        """
        self.quant_bits = given_quant_bits
        self._calculate_quant_range()

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize a float tensor.

        Computes `round(x / scale_factor + zero_point)`, clamped to the integer
        range.

        Args:
            x: Float tensor to quantize.

        Returns:
            The quantized tensor with dtype `torch.int32`, on the device of `x`.

        Raises:
            TypeError: If `x` is not a `torch.Tensor` or does not have a
                floating-point dtype.
        """
        check_dtype("x", x, *FLOAT_DTYPES)
        scale_factor = self.scale_factor.to(x.device)
        zero_point = self.zero_point.to(x.device)

        x_q = x / scale_factor + zero_point
        x_q = x_q.round_().clamp(min=self.min_quant.item(), max=self.max_quant.item())
        x_q = x_q.to(torch.int32)

        return x_q.to(x.device)

    def dequantize(self, x_q: torch.Tensor) -> torch.Tensor:
        """Map a quantized tensor back to float values.

        Computes `(x_q - zero_point) * scale_factor`.

        Args:
            x_q: Quantized tensor with dtype `torch.int32`.

        Returns:
            The dequantized tensor with dtype `torch.float32`.

        Raises:
            TypeError: If `x_q` is not a `torch.Tensor` or does not have dtype
                `torch.int32`.
        """
        zero_point = self.zero_point.to(x_q.device)
        scale_factor = self.scale_factor.to(x_q.device)
        x_q = self.math_ops.intsub(x_q, zero_point, self.quant_bits + 1)
        x_dq = x_q * scale_factor
        return x_dq.to(torch.float32).to(x_q.device)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key_names = [
            "min_float",
            "max_float",
            "min_quant",
            "max_quant",
            "scale_factor",
            "zero_point",
        ]
        for key in key_names:
            value = getattr(self, key)
            value.data = state_dict[prefix + key].data
            state_dict.pop(prefix + key)

    def __str__(self):
        return ", ".join(
            [
                f"min_float: {self.min_float.item():.6f}",
                f"max_float: {self.max_float.item():.6f}",
                f"min_quant: {int(self.min_quant.item())}",
                f"max_quant: {int(self.max_quant.item())}",
                f"scale_factor: {self.scale_factor.item():.10f}",
                f"zero_point: {int(self.zero_point.item())}",
            ]
        )


class AsymmetricSignedQParams(QParams):
    """Signed quantization parameters with a variable zero point."""

    def __init__(self, quant_bits: int, observer: BaseObserver):
        super().__init__(quant_bits, observer, is_symmetric=False, is_signed=True)


class SymmetricSignedQParams(QParams):
    """Signed quantization parameters with zero point 0."""

    def __init__(self, quant_bits: int, observer: BaseObserver):
        super().__init__(quant_bits, observer, is_symmetric=True, is_signed=True)


class AsymmetricUnsignedQParams(QParams):
    """Unsigned quantization parameters with a variable zero point."""

    def __init__(self, quant_bits: int, observer: BaseObserver):
        super().__init__(quant_bits, observer, is_symmetric=False, is_signed=False)


# BUG: The symmetric unsigned scheme is broken. `calculate_symmetric_quant_params`
#   widens the float range to `[-m, m]` with `m = max(|min_float|, |max_float|)`
#   and computes `scale = 2m / (max_quant - min_quant)`, but with zero point 0
#   and the unsigned range `[0, 254]` the codes actually represent `[0, 2m]`.
#   Negative inputs are clamped to 0 (e.g. -2.0 -> 0.0) and codes 128-254 are
#   never used, so even non-negative data only gets 7-bit precision.
#   Possible solution: support the scheme for non-negative data only (e.g.
#   after ReLU), where the fixed zero point 0 saves the zero-point subtraction
#   in hardware:
#   - add an unsigned branch to `calculate_symmetric_quant_params` that uses
#     the float range `[0, max_float]` and `scale = max_float / max_quant`,
#   - use the integer range `[0, 2**quant_bits - 1]` in `_calculate_quant_range`
#     (dropping a code only matters for signed symmetric ranges),
#   - clamp negative inputs deliberately and document it, or raise a
#     `ValueError` in `update_quant_params` if `min_float < 0`,
#   - remove the `NotImplementedError` below and rewrite the test with
#     correct expected values.
class SymmetricUnsignedQParams(QParams):
    """Unsigned quantization parameters with zero point 0.

    Not usable yet: the scheme quantizes all negative values to 0 and uses
    only half of the integer range. Use `AsymmetricUnsignedQParams` instead.

    Raises:
        NotImplementedError: Always, until the scheme is fixed.
    """

    def __init__(self, quant_bits: int, observer: BaseObserver):
        # super().__init__(quant_bits, observer, is_symmetric=True, is_signed=False)
        raise NotImplementedError(
            "SymmetricUnsignedQParams is broken: it quantizes negative values to 0 "
            "and uses only half of the integer range. Use AsymmetricUnsignedQParams "
            "instead."
        )


class QParamsFactory(Protocol):
    """Callable that creates a `QParams` object, such as a `QParams` subclass.

    Layers take a factory instead of a `QParams` object so that every layer
    gets its own instance with its own observer.
    """

    def __call__(self, quant_bits: int, observer: BaseObserver) -> QParams: ...
