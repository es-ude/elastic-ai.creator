import pytest

from elasticai.creator.arithmetic.fxp_converter import FxpConverter, FxpParams


@pytest.mark.parametrize(
    "total_bits, frac_bits, number, check",
    [
        (2, 0, 1, '"01"'),
        (2, 1, -1, '"11"'),
        (4, 2, 1, '"0001"'),
        (4, 2, -1, '"1111"'),
        (4, 2, -3, '"1101"'),
        (8, 4, -98, '"10011110"'),
        (8, 4, 35, '"00100011"'),
    ],
)
def test_convert_integer_to_binary_vhdl(
    total_bits: int, frac_bits: int, number: int, check: str
):
    rslt = FxpConverter(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    ).integer_to_binary_string_vhdl(number)
    assert rslt == check


@pytest.mark.parametrize(
    "total_bits, frac_bits, number, check",
    [
        (2, 0, 1, 'X"1"'),
        (2, 1, -1, 'X"3"'),
        (4, 2, 1, 'X"1"'),
        (4, 2, -1, 'X"F"'),
        (4, 2, -3, 'X"D"'),
        (8, 4, -99, 'X"9D"'),
        (8, 4, 35, 'X"23"'),
    ],
)
def test_convert_integer_to_hex_vhdl(
    total_bits: int, frac_bits: int, number: int, check: str
):
    rslt = FxpConverter(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits)
    ).integer_to_hex_string_vhdl(number)
    assert rslt == check


@pytest.mark.parametrize(
    "total_bits, frac_bits, number, check",
    [
        (2, 0, 1.0, '"01"'),
        (2, 1, -1.0, '"10"'),
        (4, 2, 1.0, '"0100"'),
        (4, 2, -1.0, '"1100"'),
        (4, 2, -0.75, '"1101"'),
        (8, 4, -4.6, '"10110110"'),
        (8, 4, 2.4, '"00100110"'),
    ],
)
def test_convert_rational_to_binary_vhdl(
    total_bits: int, frac_bits: int, number: int, check: str
):
    rslt = FxpConverter(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits)
    ).rational_to_binary_string_vhdl(number)
    assert rslt == check


@pytest.mark.parametrize(
    "total_bits, frac_bits, number, check",
    [
        (2, 0, 1.0, 'X"1"'),
        (2, 1, -1.0, 'X"2"'),
        (4, 2, 1.0, 'X"4"'),
        (4, 2, -1.0, 'X"C"'),
        (4, 2, -0.75, 'X"D"'),
        (8, 4, -4.6, 'X"B6"'),
        (8, 4, 2.4, 'X"26"'),
    ],
)
def test_convert_rational_to_hex_vhdl(
    total_bits: int, frac_bits: int, number: int, check: str
):
    rslt = FxpConverter(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits)
    ).rational_to_hex_string_vhdl(number)
    assert rslt == check


@pytest.mark.parametrize(
    "total_bits, frac_bits, number, check",
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
    total_bits: int, frac_bits: int, number: int, check: str
):
    rslt = FxpConverter(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits)
    ).integer_to_binary_string_verilog(number)
    assert rslt == check


@pytest.mark.parametrize(
    "total_bits, frac_bits, number, check",
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
    total_bits: int, frac_bits: int, number: int, check: str
):
    rslt = FxpConverter(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits)
    ).integer_to_decimal_string_verilog(number)
    assert rslt == check


@pytest.mark.parametrize(
    "total_bits, frac_bits, number, check",
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
    total_bits: int, frac_bits: int, number: int, check: str
):
    rslt = FxpConverter(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits)
    ).integer_to_hex_string_verilog(number)
    assert rslt == check


@pytest.mark.parametrize(
    "total_bits, frac_bits, number, check",
    [
        (2, 0, 1, "2'b01"),
        (2, 1, -0.5, "2'b11"),
        (4, 2, 0.25, "4'b0001"),
        (4, 2, -0.25, "4'b1111"),
        (4, 2, -0.75, "4'b1101"),
        (8, 4, 1.75, "8'b00011100"),
        (8, 6, -0.4345, "8'b11100100"),
    ],
)
def test_convert_rational_to_binary_verilog(
    total_bits: int, frac_bits: int, number: int, check: str
):
    rslt = FxpConverter(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits)
    ).rational_to_binary_string_verilog(number)
    assert rslt == check


@pytest.mark.parametrize(
    "total_bits, frac_bits, number, check",
    [
        (2, 0, 1, "2'd1"),
        (2, 1, -0.5, "2'd3"),
        (4, 2, 0.25, "4'd1"),
        (4, 2, -0.25, "4'd15"),
        (4, 2, -0.75, "4'd13"),
        (8, 4, 1.75, "8'd28"),
        (8, 6, -0.4345, "8'd228"),
    ],
)
def test_convert_rational_to_decimal_verilog(
    total_bits: int, frac_bits: int, number: float, check: str
):
    rslt = FxpConverter(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits)
    ).rational_to_decimal_string_verilog(number)
    assert rslt == check


@pytest.mark.parametrize(
    "total_bits, frac_bits, number, check",
    [
        (2, 0, 1, "2'h1"),
        (2, 1, -0.5, "2'h3"),
        (4, 2, 0.25, "4'h1"),
        (4, 2, -0.25, "4'hF"),
        (4, 2, -0.75, "4'hD"),
        (8, 4, 1.75, "8'h1C"),
        (8, 6, -0.4345, "8'hE4"),
    ],
)
def test_convert_rational_to_hex_verilog(
    total_bits: int, frac_bits: int, number: float, check: str
):
    rslt = FxpConverter(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits)
    ).rational_to_hex_string_verilog(number)
    assert rslt == check


@pytest.mark.parametrize(
    "total_bits, frac_bits, signed, number, expected",
    [
        (2, 0, True, "01", 1),
        (2, 0, False, "11", 3),
        (2, 1, True, "10", -2),
        (4, 2, True, "0100", 4),
        (4, 2, True, "1100", -4),
        (4, 0, True, "1100", -4),
        (4, 0, False, "1100", 12),
        (4, 2, True, "1101", -3),
        (8, 4, True, "10110110", -74),
        (8, 4, True, "00100110", 38),
    ],
)
def test_convert_binary_vhdl_to_integer(
    total_bits: int, frac_bits: int, signed: bool, number: str, expected: str
):
    rslt = FxpConverter(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=signed)
    ).binary_to_integer(number)
    assert rslt == expected


@pytest.mark.parametrize(
    "total_bits, frac_bits, signed, number, expected",
    [
        (2, 0, True, '"01"', 1.0),
        (2, 0, False, '"11"', 3.0),
        (2, 1, True, '"10"', -1.0),
        (4, 2, True, '"0100"', 1.0),
        (4, 2, True, "4'b0100", 1.0),
        (4, 2, True, '"1100"', -1.0),
        (4, 2, True, '"1101"', -0.75),
        (4, 2, True, "4'b1101", -0.75),
        (8, 4, True, '"10110110"', -4.625),
        (8, 4, True, '"00100110"', 2.375),
        (8, 4, True, "8'b00100110", 2.375),
    ],
)
def test_convert_binary_vhdl_to_float(
    total_bits: int, frac_bits: int, signed: bool, number: str, expected: str
):
    rslt = FxpConverter(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=signed)
    ).binary_to_rational(number)
    assert rslt == expected
