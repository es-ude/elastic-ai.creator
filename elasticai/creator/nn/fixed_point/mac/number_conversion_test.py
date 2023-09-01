from fixed_point.mac.number_conversion import bits_to_rational, integer_to_bits


def test_001_to_0_5():
    pattern = "001"
    assert 0.5 == bits_to_rational(pattern, frac_bits=1)


def test_001_to_0_25():
    pattern = "001"
    assert 0.25 == bits_to_rational(pattern, frac_bits=2)


def test_100_to_minus_4():
    assert -4 == bits_to_rational("100", frac_bits=0)


def test_100_to_minus_2():
    assert -2 == bits_to_rational("100", frac_bits=1)


def test_110_to_minus_1():
    pattern = "110"
    assert -1 == bits_to_rational(pattern, frac_bits=1)


def test_1_to_minus_1():
    pattern = "1"
    assert -1 == bits_to_rational(pattern, frac_bits=0)


def test_10_to_minus_2():
    pattern = "10"
    assert -2 == bits_to_rational(pattern, frac_bits=0)


def test_01_to_1():
    pattern = "01"
    assert 1 == bits_to_rational(pattern, frac_bits=0)


def test_1_to_01():
    assert "01" == integer_to_bits(number=1, total_bits=2)


def test_2_to_10():
    assert "10" == integer_to_bits(number=2, total_bits=2)


def test_minus_5_to_1011():
    assert "1011" == integer_to_bits(number=-5, total_bits=4)


def test_minus_8_to_1000():
    assert "1000" == integer_to_bits(number=-8, total_bits=4)
