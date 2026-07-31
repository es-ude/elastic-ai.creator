from typing import TypeVar, cast, overload

import torch

from elasticai.creator.arithmetic._int_arith_protocol import IntArithmetic

from .params import ConvertableToIntegerValues, LinearParams

T = TypeVar("T", bound="ConvertableToIntegerValues")


class LinearArithmetic(IntArithmetic):
    def __init__(self, params: LinearParams):
        self._config = params

    @property
    def config(self) -> LinearParams:
        return self._config

    @property
    def total_bits(self) -> int:
        return self.config.total_bits

    def integer_out_of_bounds(self, number):
        return self._config._check_integer_out_of_bounds(number)

    @property
    def minimum_as_integer(self) -> int:
        return self.config.minimum_as_integer

    @property
    def maximum_as_integer(self) -> int:
        return self.config.maximum_as_integer

    @property
    def minimum_as_rational(self) -> float:
        return self.config.minimum_as_rational

    @property
    def maximum_as_rational(self) -> float:
        return self.config.maximum_as_rational

    ##################################################
    ## CUT

    ### TO INT

    @overload
    def cut_as_integer(self, number: float | int) -> int:
        """Cutting the input number to integer directly (more like in hardware)"""
        ...

    @overload
    def cut_as_integer(self, number: list[float | int]) -> list[int]:
        """Cutting the input number to integer directly (more like in hardware)"""
        ...

    @overload
    def cut_as_integer(self, number: list[list[float | int]]) -> list[list[int]]:
        """Cutting the input number to integer directly (more like in hardware)"""
        ...

    @overload
    def cut_as_integer(self, number: T) -> T:
        """Cutting the input number to integer directly (more like in hardware)"""
        ...

    def cut_as_integer(self, number: float | int | list | T) -> int | list | T:
        if isinstance(number, ConvertableToIntegerValues):
            return (
                (number.float() / self._config.scale_factor).int()
                + self._config.zero_point
            ).float()
        elif isinstance(number, int):
            return int(number / self._config.scale_factor) + self._config.zero_point
        elif isinstance(number, float):
            return int(number / self._config.scale_factor) + self._config.zero_point
        elif isinstance(number, list):
            return [self.cut_as_integer(n) for n in number]

    ### AS RATIONAL

    @overload
    def cut_as_rational(self, number: float) -> float: ...

    @overload
    def cut_as_rational(self, number: list[float]) -> list[float]: ...

    @overload
    def cut_as_rational(self, number: list[list[float]]) -> list[list[float]]: ...

    @overload
    def cut_as_rational(self, number: T) -> T: ...

    def cut_as_rational(self, number: float | int | list | T) -> float | int | list | T:
        if isinstance(number, ConvertableToIntegerValues):
            scale_factor: float = self._config.scale_factor
            zero_point: int = self._config.zero_point
            return cast(T, (self.cut_as_integer(number) - zero_point) * scale_factor)
        elif isinstance(number, (float, int)):
            return self._config.scale_factor * (
                self.cut_as_integer(number) - self._config.zero_point
            )
        elif isinstance(number, list):
            return [self.cut_as_rational(n) for n in number]

    ##################################################
    ## ROUND

    ### TO INT

    @overload
    def round_to_integer(self, number: float | int) -> int:
        """Mathematical Round function for number"""
        ...

    @overload
    def round_to_integer(self, number: T) -> T:
        """Mathematical Round function for number"""
        ...

    @overload
    def round_to_integer(self, number: list[float | int]) -> list[int]:
        """Mathematical Round function for number"""
        ...

    @overload
    def round_to_integer(self, number: list[T]) -> list[T]:
        """Mathematical Round function for number"""
        ...

    def round_to_integer(self, number: float | int | list | T) -> int | list | T:
        if isinstance(number, ConvertableToIntegerValues):
            return (
                (number.float() / self._config.scale_factor).round()
                + self._config.zero_point
            ).float()
        elif isinstance(number, int):
            return round(number / self._config.scale_factor) + self._config.zero_point
        elif isinstance(number, float):
            return int(
                round(number / self._config.scale_factor) + self._config.zero_point
            )
        elif isinstance(number, list):
            return [self.round_to_integer(n) for n in number]

    ### AS RATIONAL

    @overload
    def round_to_rational(self, number: float) -> float: ...

    @overload
    def round_to_rational(self, number: list[float]) -> list[float]: ...

    @overload
    def round_to_rational(self, number: list[list[float]]) -> list[list[float]]: ...

    @overload
    def round_to_rational(self, number: T) -> T: ...

    def round_to_rational(
        self, number: float | int | list | T
    ) -> float | int | list | T:
        if isinstance(number, ConvertableToIntegerValues):
            scale_factor: float = self._config.scale_factor
            zero_point: int = self._config.zero_point
            return cast(T, (self.round_to_integer(number) - zero_point) * scale_factor)
        elif isinstance(number, (float, int)):
            return self._config.scale_factor * (
                self.round_to_integer(number) - self._config.zero_point
            )
        elif isinstance(number, list):
            return [self.round_to_rational(n) for n in number]

    ##################################################
    ## CLAMP

    @overload
    def clamp(self, number: int) -> int: ...

    @overload
    def clamp(self, number: float) -> float: ...

    @overload
    def clamp(self, number: T) -> T: ...

    def clamp(self, number: int | float | T) -> int | float | T:
        if isinstance(number, ConvertableToIntegerValues):
            return number.clamp(
                min=self._config.minimum_as_integer, max=self._config.maximum_as_integer
            )
        elif isinstance(number, int):
            # return number or minimum_as_integer or maximum_as_integer, whichever is in bounds
            return max(
                self._config.minimum_as_integer,
                min(self._config.maximum_as_integer, number),
            )
        elif isinstance(number, float):
            # return number or minimum_as_rational or maximum_as_rational, whichever is in bounds
            return max(
                self._config.minimum_as_rational,
                min(self._config.maximum_as_rational, number),
            )

    ##################################################
    ## DEQUANTIZE

    @overload
    def as_rational(self, number: int) -> float: ...

    @overload
    def as_rational(self, number: T) -> T: ...

    def as_rational(self, number: int | T) -> float | T:
        if isinstance(number, ConvertableToIntegerValues):
            scale_factor: float = self._config.scale_factor
            zero_point: int = self._config.zero_point
            return cast(T, (number - zero_point) * scale_factor)
        elif isinstance(number, int):
            return self._config.scale_factor * (number - self._config.zero_point)

    ##################################################
    ## TWO COMPLEMENT

    @overload
    def to_twos(self, number: int) -> int: ...

    @overload
    def to_twos(self, number: float) -> int: ...

    @overload
    def to_twos(self, number: T) -> T: ...

    def to_twos(self, number: int | float | T) -> int | T:
        # TODO: implement two's complement conversion for float and T types
        raise NotImplementedError(
            "Two's complement conversion is not implemented for float and T types yet."
        )

    @overload
    def is_power_of_2(self, number: T) -> T: ...

    @overload
    def is_power_of_2(self, number: int) -> bool: ...

    def is_power_of_2(self, number: int | T) -> bool | T:
        # TODO: implement power of 2 check for T types
        raise NotImplementedError(
            "Power of 2 check is not implemented for T types yet."
        )


class CutToInteger(torch.autograd.Function):
    """Straight-through estimator for hard (truncating) integer quantization.

    Wraps :meth:`LinearArithmetic.cut_as_integer` in an autograd ``Function``
    so the (non-differentiable) quantize -> dequantize round trip can be used
    inside a training graph. The forward pass cuts ``number`` to its integer
    fixed-point representation and immediately maps it back to the
    corresponding rational value, so callers see a "fake quantized" tensor
    with the same shape/dtype as the input. The backward pass is a
    straight-through estimator: gradients flow through unchanged as if this
    function were the identity, and no gradient is produced for ``config``.
    """

    @staticmethod
    def forward(
        ctx,
        number: float | int | list | T,
        config: LinearArithmetic,
    ):
        """Cuts ``number`` to integer and maps it back to a rational value.

        Args:
            ctx: Autograd context (unused, kept for the ``Function`` interface).
            number: Value(s) to quantize; anything accepted by
                :meth:`LinearArithmetic.cut_as_integer`.
            config: Arithmetic configuration used to cut and dequantize
                ``number``.

        Returns:
            The dequantized ("fake quantized") value(s), same container type
            as ``number``.

        Raises:
            ValueError: If any cut integer value falls outside the
                representable integer range of ``config``.
        """
        fxp_ints = config.cut_as_integer(number)
        out_of_bounds = config.integer_out_of_bounds(fxp_ints)
        if torch.any(out_of_bounds):
            raise ValueError("Cannot quantize tensor. Values out of bounds.")
        return config.as_rational(fxp_ints)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Passes gradients through unchanged (straight-through estimator).

        Args:
            ctx: Autograd context (unused).
            *grad_outputs: Gradients w.r.t. the outputs of :meth:`forward`.

        Returns:
            The incoming gradients followed by ``None`` for the
            non-differentiable ``config`` argument.
        """
        return *grad_outputs, None


class RoundToInteger(torch.autograd.Function):
    """Straight-through estimator for rounding integer quantization.

    Same purpose as :class:`CutToInteger`, but rounds ``number`` to the
    nearest representable integer (via
    :meth:`LinearArithmetic.round_to_integer`) instead of truncating. Unlike
    ``CutToInteger``, the forward pass returns the integer representation
    directly rather than mapping it back to a rational value.
    """

    @staticmethod
    def forward(ctx, number: float | int | list | T, config: LinearArithmetic):
        """Rounds ``number`` to its integer fixed-point representation.

        Args:
            ctx: Autograd context (unused, kept for the ``Function`` interface).
            number: Value(s) to round; anything accepted by
                :meth:`LinearArithmetic.round_to_integer`.
            config: Arithmetic configuration used to round and bounds-check
                ``number``.

        Returns:
            The rounded integer value(s), same container type as ``number``.

        Raises:
            ValueError: If any rounded integer value falls outside the
                representable integer range of ``config``.
        """
        fxp_ints = config.round_to_integer(number)
        out_of_bounds = config.integer_out_of_bounds(fxp_ints)
        if torch.any(out_of_bounds):
            raise ValueError("Cannot quantize tensor. Values out of bounds.")
        return fxp_ints

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Passes gradients through unchanged (straight-through estimator).

        Args:
            ctx: Autograd context (unused).
            *grad_outputs: Gradients w.r.t. the outputs of :meth:`forward`.

        Returns:
            The incoming gradients followed by ``None`` for the
            non-differentiable ``config`` argument.
        """
        return *grad_outputs, None
