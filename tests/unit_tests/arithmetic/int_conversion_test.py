from random import randint

import pytest

from elasticai.creator.arithmetic._int_converter import int_converter
from elasticai.creator.arithmetic.fxp_params import FxpParams


@pytest.mark.parametrize(
    "total_bits, is_signed, number, check",
    [
        (2, True, 1, '"01"'),
        (2, True, -1, '"11"'),
        (4, True, 1, '"0001"'),
        (4, True, -1, '"1111"'),
        (4, True, -3, '"1101"'),
        (8, True, -98, '"10011110"'),
        (8, True, 35, '"00100011"'),
        (2, False, 1, '"01"'),
        (2, False, 3, '"11"'),
        (4, False, 1, '"0001"'),
        (4, False, 15, '"1111"'),
        (4, False, 13, '"1101"'),
        (8, False, 165, '"10100101"'),
        (8, False, 35, '"00100011"'),
    ],
)
def test_convert_integer_to_binary_vhdl(
    total_bits: int, is_signed: bool, number: int, check: str
):
    rslt = int_converter(
        total_bits=total_bits, signed=is_signed
    ).integer_to_binary_string_vhdl(number)
    assert rslt == check


@pytest.mark.parametrize(
    "total_bits, is_signed, number, check",
    [
        (2, True, 1, 'X"1"'),
        (2, True, -1, 'X"3"'),
        (4, True, 1, 'X"1"'),
        (4, True, -1, 'X"F"'),
        (4, True, -3, 'X"D"'),
        (8, True, -99, 'X"9D"'),
        (8, True, 35, 'X"23"'),
        (2, False, 1, 'X"1"'),
        (2, False, 3, 'X"3"'),
        (4, False, 1, 'X"1"'),
        (4, False, 15, 'X"F"'),
        (4, False, 3, 'X"3"'),
        (8, False, 99, 'X"63"'),
        (8, False, 35, 'X"23"'),
    ],
)
def test_convert_integer_to_hex_vhdl(
    total_bits: int, is_signed: bool, number: int, check: str
):
    rslt = int_converter(
        total_bits=total_bits, signed=is_signed
    ).integer_to_hex_string_vhdl(number)
    assert rslt == check


@pytest.mark.parametrize(
    "total_bits, is_signed, number, check",
    [
        (2, 0, 1, "2'b01"),
        (2, 1, -1, "2'b11"),
        (4, 2, 1, "4'b0001"),
        (4, 2, -1, "4'b1111"),
        (4, 2, -3, "4'b1101"),
        (8, 4, 103, "8'b01100111"),
        (8, 6, -7, "8'b11111001"),
    ],
)
def test_convert_integer_to_binary_verilog(
    total_bits: int, is_signed: bool, number: int, check: str
):
    rslt = int_converter(
        total_bits=total_bits, signed=is_signed
    ).integer_to_binary_string_verilog(number)
    assert rslt == check


@pytest.mark.parametrize(
    "total_bits, is_signed, number, check",
    [
        (2, 0, 1, "2'd1"),
        (2, 1, -1, "2'd3"),
        (4, 2, 1, "4'd1"),
        (4, 2, -1, "4'd15"),
        (4, 2, -3, "4'd13"),
        (8, 4, -101, "8'd155"),
        (8, 4, 35, "8'd35"),
    ],
)
def test_convert_integer_to_decimal_verilog(
    total_bits: int, is_signed: bool, number: int, check: str
):
    rslt = int_converter(
        total_bits=total_bits, signed=is_signed
    ).integer_to_decimal_string_verilog(number)
    assert rslt == check


@pytest.mark.parametrize(
    "total_bits, is_signed, number, check",
    [
        (2, 0, 1, "2'h1"),
        (2, 1, -1, "2'h3"),
        (4, 2, 1, "4'h1"),
        (4, 2, -1, "4'hF"),
        (4, 2, -3, "4'hD"),
        (8, 4, -101, "8'h9B"),
        (8, 4, 35, "8'h23"),
    ],
)
def test_convert_integer_to_hex_verilog(
    total_bits: int, is_signed: bool, number: int, check: str
):
    rslt = int_converter(
        total_bits=total_bits, signed=is_signed
    ).integer_to_hex_string_verilog(number)
    assert rslt == check


@pytest.mark.parametrize(
    "total_bits, is_signed, number, expected",
    [
        (2, True, "01", 1),
        (2, False, "11", 3),
        (2, True, "10", -2),
        (4, True, "0100", 4),
        (4, True, "1100", -4),
        (4, True, "1100", -4),
        (4, False, "1100", 12),
        (4, True, "1101", -3),
        (8, True, "10110110", -74),
        (8, True, "00100110", 38),
    ],
)
def test_convert_binary_vhdl_to_integer(
    total_bits: int, is_signed: bool, number: str, expected: str
):
    rslt = int_converter(total_bits=total_bits, signed=is_signed).binary_to_integer(
        number
    )
    assert rslt == expected


@pytest.mark.parametrize(
    "total_bits, is_signed, number, expected",
    [
        (2, True, "01", 1),
        (2, True, "'b01", 1),
        (2, True, "2'b01", 1),
        (2, False, "11", 3),
        (2, False, "'b11", 3),
        (2, False, "2'b11", 3),
        (8, True, "10110110", -74),
        (8, True, "'b10110110", -74),
        (8, True, "8'b10110110", -74),
        (8, False, "00100110", 38),
        (8, False, "'b00100110", 38),
        (8, False, "8'b00100110", 38),
    ],
)
def test_convert_binary_string_to_integer(
    total_bits: int, is_signed: bool, number: str, expected: str
):
    rslt = int_converter(total_bits=total_bits, signed=is_signed).binary_to_integer(
        number
    )
    assert rslt == expected


@pytest.mark.parametrize("total_bits", [2, 4, 8, 16])
@pytest.mark.parametrize("is_signed", [False, True])
def test_convert_binary_string_to_integer_back(total_bits: int, is_signed: bool):
    sets = FxpParams(total_bits=total_bits, frac_bits=0, signed=is_signed)
    config = int_converter(total_bits=total_bits, signed=is_signed)

    for _ in range(8):
        val = randint(a=sets.minimum_as_integer, b=sets.maximum_as_integer)
        bin_vhdl_out = config.integer_to_binary_string_vhdl(val)
        val_vhdl_out = config.binary_to_integer(bin_vhdl_out)
        assert val == val_vhdl_out

        bin_verilog_out = config.integer_to_binary_string_verilog(val)
        val_verilog_out = config.binary_to_integer(bin_verilog_out)
        assert val == val_verilog_out
