from random import randint

import pytest

from elasticai.creator.arithmetic._int_converter import int_converter
from elasticai.creator.arithmetic.fxp_params import FxpParams


@pytest.mark.parametrize(
    "total_bits, is_signed, number, check",
    [
        (8, True, -98, '"10011110"'),
        (8, True, 35, '"00100011"'),
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
        (8, True, -99, 'X"9D"'),
        (8, True, 35, 'X"23"'),
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
    "total_bits, is_signed, numbers, check",
    [
        (4, True, [-7, -4, -1, 2, 5], "{4'b1001, 4'b1100, 4'b1111, 4'b0010, 4'b0101}"),
        (4, False, [9, 12, 15, 2, 5], "{4'b1001, 4'b1100, 4'b1111, 4'b0010, 4'b0101}"),
    ],
)
def test_convert_integer_to_binary_array_verilog(
    total_bits: int, is_signed: bool, numbers: list[int], check: str
):
    rslt = int_converter(
        total_bits=total_bits, signed=is_signed
    ).integer_to_binary_string_array_verilog(numbers)
    assert rslt == check


@pytest.mark.parametrize(
    "total_bits, is_signed, number, check",
    [
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
    "total_bits, is_signed, numbers, check",
    [
        (4, True, [-7, -4, -1, 2, 5], "{4'd9, 4'd12, 4'd15, 4'd2, 4'd5}"),
        (4, False, [9, 12, 15, 2, 5], "{4'd9, 4'd12, 4'd15, 4'd2, 4'd5}"),
    ],
)
def test_convert_integer_to_decimal_array_verilog(
    total_bits: int, is_signed: bool, numbers: list[int], check: str
):
    rslt = int_converter(
        total_bits=total_bits, signed=is_signed
    ).integer_to_decimal_string_array_verilog(numbers)
    assert rslt == check


@pytest.mark.parametrize(
    "total_bits, is_signed, number, check",
    [
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
    "total_bits, is_signed, numbers, check",
    [
        (4, True, [-7, -4, -1, 2, 5], "{4'h9, 4'hC, 4'hF, 4'h2, 4'h5}"),
        (4, False, [9, 12, 15, 2, 5], "{4'h9, 4'hC, 4'hF, 4'h2, 4'h5}"),
    ],
)
def test_convert_integer_to_hex_array_verilog(
    total_bits: int, is_signed: bool, numbers: list[int], check: str
):
    rslt = int_converter(
        total_bits=total_bits, signed=is_signed
    ).integer_to_hex_string_array_verilog(numbers)
    assert rslt == check


@pytest.mark.parametrize(
    "total_bits, is_signed, number, expected",
    [
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
        (8, True, "10110110", -74),
        (8, True, "'b10110110", -74),
        (8, True, "8'b10110110", -74),
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
