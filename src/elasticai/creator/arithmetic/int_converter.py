from dataclasses import dataclass
from math import ceil
from typing import cast

from elasticai.creator.arithmetic.int_arithmetic import IntArithmetic
from elasticai.creator.arithmetic.int_params import (
    ConvertableToIntegerValues,
    IntParams,
    T,
)


@dataclass
class IntConverter:
    def __init__(self, fxp_params: IntParams):
        self._config = fxp_params

    @property
    def total_bits(self) -> int:
        return self._config.total_bits

    def _round_to_integer(self, number: float | T) -> int | T:
        return IntArithmetic(self._config).round_to_integer(number)

    def _integer_to_twos(self, number):
        if isinstance(number, ConvertableToIntegerValues):
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

    def integer_to_decimal_string_verilog(self, number: int) -> str:
        twos = self._integer_to_twos(number)
        return f"{self._config.total_bits}'d{twos}"

    def integer_to_hex_string_verilog(self, number: int) -> str:
        twos = self._integer_to_twos(number)
        return (
            f"{self._config.total_bits}'h{twos:0{ceil(self._config.total_bits / 4)}X}"
        )

    def binary_string_to_integer(self, binary: str) -> int:
        format_binary = binary.replace('"', "").replace(" ", "").split("b")[-1]
        return int(format_binary, 2) - (
            (1 << self._config.total_bits)
            if self._config.signed and format_binary[0] == "1"
            else 0
        )
