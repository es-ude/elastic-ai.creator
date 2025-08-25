from dataclasses import dataclass
from math import ceil
from typing import cast

from elasticai.creator.arithmetic.fxp_arithmetic import FxpArithmetic
from elasticai.creator.arithmetic.fxp_params import (
    ConvertableToFixedPointValues,
    FxpParams,
    T,
)


@dataclass
class FxpConverter(FxpParams):
    def __init__(self, total_bits: int, frac_bits: int, signed: bool = True):
        self.total_bits = total_bits
        self.frac_bits = frac_bits
        self.signed = signed

    def _round_to_integer(self, number: float | int | T) -> int | T:
        return FxpArithmetic(
            total_bits=self.total_bits, frac_bits=self.frac_bits, signed=self.signed
        ).round_to_integer(number)

    def _integer_to_twos(self, number: int | float | T) -> int:
        if isinstance(number, ConvertableToFixedPointValues):
            return int(self._convert_integer_to_twos(cast(T, number)))
        else:
            return self._convert_integer_to_twos(number)

    def _convert_integer_to_twos(self, number: int | float | T) -> int | float | T:
        if self.integer_out_of_bounds(number):
            raise ValueError(
                f"Value '{number}' cannot be represented with {self.total_bits} bits."
            )
        return (1 << self.total_bits) + number if number < 0 else number

    def integer_to_binary_string_vhdl(self, number: int) -> str:
        twos = self._integer_to_twos(number)
        return f'"{twos:0{self.total_bits}b}"'

    def integer_to_hex_string_vhdl(self, number: int) -> str:
        twos = self._integer_to_twos(number)
        return f'X"{twos:0{ceil(self.total_bits / 4)}X}"'

    def integer_to_binary_string_verilog(self, number: int) -> str:
        twos = self._integer_to_twos(number)
        return f"{self.total_bits}'b{twos:0{self.total_bits}b}"

    def integer_to_decimal_string_verilog(self, number: int) -> str:
        twos = self._integer_to_twos(number)
        return f"{self.total_bits}'d{twos}"

    def integer_to_hex_string_verilog(self, number: int) -> str:
        twos = self._integer_to_twos(number)
        return f"{self.total_bits}'h{twos:0{ceil(self.total_bits / 4)}X}"

    def rational_to_binary_string_vhdl(self, number: float) -> str:
        fxp = self._round_to_integer(number)
        return self.integer_to_binary_string_vhdl(fxp)

    def rational_to_hex_string_vhdl(self, number: float) -> str:
        fxp = self._round_to_integer(number)
        return self.integer_to_hex_string_vhdl(fxp)

    def rational_to_binary_string_verilog(self, number: float) -> str:
        fxp = self._round_to_integer(number)
        return self.integer_to_binary_string_verilog(fxp)

    def rational_to_decimal_string_verilog(self, number: float) -> str:
        fxp = self._round_to_integer(number)
        return self.integer_to_decimal_string_verilog(fxp)

    def rational_to_hex_string_verilog(self, number: float) -> str:
        fxp = self._round_to_integer(number)
        return self.integer_to_hex_string_verilog(fxp)

    def binary_to_integer(self, binary: str) -> int:
        format_binary = binary.replace('"', "").replace(" ", "").split("b")[-1]
        return int(format_binary, 2) - (
            (1 << self.total_bits) if self.signed and format_binary[0] == "1" else 0
        )

    def binary_to_rational(self, binary: str) -> float:
        int_val = self.binary_to_integer(binary)
        return int_val * self.minimum_step_as_rational

    def decimal_to_integer(self, binary: str) -> int:
        format_binary = binary.split("d")[-1]
        return int(format_binary) - (
            (1 << self.total_bits) if self.signed and format_binary[0] == "1" else 0
        )

    def decimal_to_rational(self, binary: str) -> float:
        int_val = self.decimal_to_integer(binary)
        return int_val * self.minimum_step_as_rational

    def hex_to_integer(self, binary: str) -> int:
        format_binary = binary.replace('"', "").replace(" ", "").split("X")[-1]
        return int(format_binary, 16) - (
            (1 << self.total_bits) if self.signed and format_binary[0] == "1" else 0
        )

    def hex_to_rational(self, binary: str) -> float:
        int_val = self.hex_to_integer(binary)
        return int_val * self.minimum_step_as_rational
