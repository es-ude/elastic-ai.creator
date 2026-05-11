from enum import Enum

from torch import (
    Size,  # type:ignore[reportPrivateImportUsage]
    Tensor,
)

from elasticai.creator.arithmetic.fxp_arithmetic import FxpArithmetic


class DeltaType(Enum):
    CONSECUTIVE = "consecutive"
    FIXED_REFERENCE = "fixed-reference"


class DeltaOperations:
    """Delta operations for fixed-point arithmetic."""

    def __init__(
        self,
        fxp_arithmetic: FxpArithmetic,
        delta_bits: int,
        delta_offset: int,
        delta_type: DeltaType = DeltaType.CONSECUTIVE,
        clamp: bool = False,
    ) -> None:
        """Delta operations for fixed-point arithmetic.

        Args:
            delta_bits: The number of bits to use for the delta values.
            fxp_arithmetic: The arithmetic to use for the fixed-point operations.
            delta_offset: The offset to use for the delta bits
            clamp: Wheter to clamp the values

        Notes for clamp_style:
            - "lsb" clamps to the least significant bit
            - "tuple[int_bits,frac_bits]" clamps using given number of integer bits and fraction bits
        """
        self._delta_bits = delta_bits
        self._fxp_arithmetic = fxp_arithmetic
        self._delta_offset = delta_offset
        self._delta_type = delta_type
        self._clamp = clamp
        self.__post_init__()

    def __post_init__(self):
        if self.delta_bits <= 0:
            raise ValueError(
                f"delta bits need to be > 0 for {self.__class__.__name__}. "
                f"You have set {self.delta_bits=}."
            )
        if self.delta_offset < 0:
            raise ValueError(
                f"delta offset need to be >= 0 for {self.__class__.__name__}. "
                f"You have set {self.delta_offset=}."
            )
        if self.delta_offset + self.delta_bits > self.total_bits:
            raise ValueError(
                f"delta offset + delta bits need to be <= total bits for {self.__class__.__name__}. "
                f"You have set {self.delta_offset=} and {self.delta_bits=}."
            )

    @property
    def delta_bits(self) -> int:
        return self._delta_bits

    @property
    def delta_offset(self) -> int:
        return self._delta_offset

    @property
    def delta_type(self) -> DeltaType:
        return self._delta_type

    @property
    def total_bits(self) -> int:
        return self._fxp_arithmetic.total_bits

    @property
    def frac_bits(self) -> int:
        return self._fxp_arithmetic.frac_bits

    @property
    def clamp(self) -> bool:
        return self._clamp

    @staticmethod
    def __delta_fixed(input: Tensor) -> tuple[Tensor, Size]:
        original_shape = input.shape
        input = input.flatten()
        input[1:] -= input[0]
        return input, original_shape

    @staticmethod
    def __delta_consecutive(input: Tensor) -> tuple[Tensor, Size]:
        original_shape = input.shape
        input = input.flatten()
        for index in reversed(range(1, len(input))):
            input[index] = input[index] - input[index - 1]
        return input, original_shape

    def _delta(self, input: Tensor) -> tuple[Tensor, Size]:
        match self.delta_type:
            case DeltaType.CONSECUTIVE:
                return DeltaOperations.__delta_consecutive(input)
            case DeltaType.FIXED_REFERENCE:
                return DeltaOperations.__delta_fixed(input)

    @staticmethod
    def __bitmask(delta_bits: int, delta_offset: int) -> int:
        bitmask = 0
        for bit_index in range(delta_bits - 1):
            bitmask |= 1 << (bit_index + delta_offset)
        return bitmask

    @staticmethod
    def __compress_without_clamping(
        t: Tensor, delta_bits: int, delta_offset: int
    ) -> Tensor:
        negatvive_indexes = t < 0
        t.abs_()
        t &= DeltaOperations.__bitmask(delta_bits, delta_offset)
        t[negatvive_indexes] *= -1
        return t

    @staticmethod
    def __compress_with_clamping(
        t: Tensor, delta_bits: int, delta_offset: int
    ) -> Tensor:
        negative_indexes = t < 0
        t.abs_()
        t.clamp_(
            min=2 ** (delta_offset) if delta_offset > 0 else 0,
            max=2 ** (delta_bits - 1 + delta_offset) - 1,
        )
        t[negative_indexes] *= -1
        return t

    def _compress(self, t: Tensor) -> Tensor:
        assert isinstance(self._delta_bits, int) and self._delta_bits > 0
        assert isinstance(self._delta_offset, int) and self._delta_offset >= 0
        assert self._delta_bits + self._delta_offset <= self.total_bits

        t_int = t.int()

        if self.clamp:
            t_int = DeltaOperations.__compress_with_clamping(
                t_int, self.delta_bits, self.delta_offset
            )
        else:
            t_int = DeltaOperations.__compress_without_clamping(
                t_int, self.delta_bits, self.delta_offset
            )

        return t.copy_(t_int.float())

    def compress(self, input: Tensor) -> Tensor:
        """Compresses the input tensor using delta encoding.

        Args:
            input: The input tensor to compress. (Floating point weights.)

        Returns:
            The compressed tensor. (Fixed Point Weights)
        """
        input = self._fxp_arithmetic.cut_as_integer(input)
        input, input_shape = self._delta(input)
        input[1:] = self._compress(input[1:])
        return input.reshape(input_shape)

    @staticmethod
    def __reverse_delta_consecutive(input: Tensor) -> tuple[Tensor, Size]:
        original_shape = input.shape
        input = input.flatten()
        for index in range(1, len(input)):
            input[index] = input[index - 1] + input[index]
        return input, original_shape

    @staticmethod
    def __reverse_delta_fixed(input: Tensor) -> tuple[Tensor, Size]:
        original_shape = input.shape
        input = input.flatten()
        input[1:] += input[0]
        return input, original_shape

    def _reverse_delta(self, input) -> tuple[Tensor, Size]:
        match self.delta_type:
            case DeltaType.CONSECUTIVE:
                return DeltaOperations.__reverse_delta_consecutive(input)
            case DeltaType.FIXED_REFERENCE:
                return DeltaOperations.__reverse_delta_fixed(input)

    def inflate(self, input: Tensor) -> Tensor:
        """Inflates the input tensor using delta encoding.

        Args:
            input: The input tensor to inflate. (Fixed point weights.)

        Returns:
            The inflated tensor. (Floating point weights.)
        """
        input, input_shape = self._reverse_delta(input)
        input = self._fxp_arithmetic.as_rational(input)
        input = self._fxp_arithmetic.clamp(input)
        return input.reshape(input_shape)
