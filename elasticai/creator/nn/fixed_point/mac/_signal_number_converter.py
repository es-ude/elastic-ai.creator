from functools import partial
from typing import Iterable

from fixed_point.mac.number_conversion import (
    bits_to_rational,
    integer_to_bits,
    rational_to_bits,
)


class SignalNumberConverter:
    """Convert python input arguments to corresponding signal bit vectors for vhdl or verilog"""

    def __init__(self, frac_bits, total_bits):
        self._frac_bits = frac_bits
        self._total_bits = total_bits
        self._convert_to_bit_pattern = partial(integer_to_bits, total_bits=total_bits)

    def _logic_signal(self, value):
        return f"{value}"

    def _convert_float_to_bit_pattern(self, value):
        if isinstance(value, Iterable):
            return list(map(self._convert_float_to_bit_pattern, value))
        return rational_to_bits(
            value, total_bits=self._total_bits, frac_bits=self._frac_bits
        )

    def _clock_cycle(self, values):
        values = self._convert_float_to_bit_pattern(values)
        return [["0", "0"] + values, ["0", "1"] + values]

    def _reset(self):
        values = self._convert_float_to_bit_pattern([0, 0])
        return [["1", "0"] + values, ["0", "1"] + values]

    def to_signals(self, inputs):
        result = self._reset()
        for x1, x2 in inputs:
            result.extend(self._clock_cycle((x1, x2)))
        result.extend(self._reset())
        return result

    def to_numbers(self, signals):
        undefined = "U" * self._total_bits
        result = []
        for signal in signals:
            if signal == undefined:
                result.append(float("NaN"))
            else:
                result.append(bits_to_rational(signal, self._frac_bits))
        return result
