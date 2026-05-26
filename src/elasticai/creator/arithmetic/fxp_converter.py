from dataclasses import dataclass
from math import ceil
from typing import TypeVar, cast

from .fxp_arithmetic import FxpArithmetic
from .fxp_params import (
    ConvertableToFixedPointValues,
    FxpParams,
)

T = TypeVar("T", bound=ConvertableToFixedPointValues)


@dataclass
class FxpConverter:
    def __init__(self, fxp_params: FxpParams):
        self._config = fxp_params

    @property
    def total_bits(self) -> int:
        return self._config.total_bits

    @property
    def frac_bits(self) -> int:
        return self._config.frac_bits

    def _round_to_integer(self, number: float | T) -> int | T:
        return FxpArithmetic(self._config).round_to_integer(number)

    def _integer_to_twos(self, number):
        if isinstance(number, ConvertableToFixedPointValues):
            return int(self._convert_integer_to_twos(cast(T, number)))
        else:
            return self._convert_integer_to_twos(number)

    def _convert_integer_to_twos(self, number):
        if self._config.integer_out_of_bounds(number):
            raise ValueError(
                f"Value '{number}' cannot be represented with {self._config.total_bits} bits."
            )
        return ((1 << self._config.total_bits) + number) if number < 0 else number

    def integer_to_binary_string_vhdl(self, number: int) -> str:
        twos = self._integer_to_twos(number)
        return f'"{twos:0{self._config.total_bits}b}"'

    def integer_to_hex_string_vhdl(self, number: int) -> str:
        twos = self._integer_to_twos(number)
        return f'X"{twos:0{ceil(self._config.total_bits / 4)}X}"'

    def integer_to_binary_string_verilog(self, number: int) -> str:
        twos = self._integer_to_twos(number)
        return f"{self._config.total_bits}'b{twos:0{self._config.total_bits}b}"

    def integer_to_binary_string_array_verilog(self, numbers: list[int]) -> str:
        return (
            "{"
            + ", ".join([self.integer_to_binary_string_verilog(val) for val in numbers])
            + "}"
        )

    def integer_to_decimal_string_verilog(self, number: int) -> str:
        twos = self._integer_to_twos(number)
        return f"{self._config.total_bits}'d{twos}"

    def integer_to_decimal_string_array_verilog(self, numbers: list[int]) -> str:
        return (
            "{"
            + ", ".join(
                [self.integer_to_decimal_string_verilog(val) for val in numbers]
            )
            + "}"
        )

    def integer_to_hex_string_verilog(self, number: int) -> str:
        twos = self._integer_to_twos(number)
        return (
            f"{self._config.total_bits}'h{twos:0{ceil(self._config.total_bits / 4)}X}"
        )

    def integer_to_hex_string_array_verilog(self, numbers: list[int]) -> str:
        return (
            "{"
            + ", ".join([self.integer_to_hex_string_verilog(val) for val in numbers])
            + "}"
        )

    def rational_to_binary_string_vhdl(self, number: float) -> str:
        fxp = self._round_to_integer(number)
        return self.integer_to_binary_string_vhdl(fxp)

    def rational_to_hex_string_vhdl(self, number: float) -> str:
        fxp = self._round_to_integer(number)
        return self.integer_to_hex_string_vhdl(fxp)

    def rational_to_binary_string_verilog(self, number: float) -> str:
        fxp = self._round_to_integer(number)
        return self.integer_to_binary_string_verilog(fxp)

    def rational_to_binary_string_array_verilog(self, numbers: list[float]) -> str:
        fxp = [self._round_to_integer(val) for val in numbers]
        return self.integer_to_binary_string_array_verilog(fxp)

    def rational_to_decimal_string_verilog(self, number: float) -> str:
        fxp = self._round_to_integer(number)
        return self.integer_to_decimal_string_verilog(fxp)

    def rational_to_decimal_string_array_verilog(self, numbers: list[float]) -> str:
        fxp = [self._round_to_integer(val) for val in numbers]
        return self.integer_to_decimal_string_array_verilog(fxp)

    def rational_to_hex_string_verilog(self, number: float) -> str:
        fxp = self._round_to_integer(number)
        return self.integer_to_hex_string_verilog(fxp)

    def rational_to_hex_string_array_verilog(self, numbers: list[float]) -> str:
        fxp = [self._round_to_integer(val) for val in numbers]
        return self.integer_to_hex_string_array_verilog(fxp)

    def binary_to_integer(self, binary: str) -> int:
        format_binary = binary.replace('"', "").replace(" ", "").split("b")[-1]
        return int(format_binary, 2) - (
            (1 << self._config.total_bits)
            if self._config.signed and format_binary[0] == "1"
            else 0
        )

    def binary_to_rational(self, binary: str) -> float:
        int_val = self.binary_to_integer(binary)
        return int_val * self._config.minimum_step_as_rational
