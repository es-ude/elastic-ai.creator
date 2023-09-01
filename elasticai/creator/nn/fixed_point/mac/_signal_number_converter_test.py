import math

from fixed_point.mac._signal_number_converter import SignalNumberConverter


def test_reset_two_times_without_input():
    generator = SignalNumberConverter(total_bits=4, frac_bits=2)
    expected = [
        ["1", "0", "0000", "0000"],
        ["0", "1", "0000", "0000"],
        ["1", "0", "0000", "0000"],
        ["0", "1", "0000", "0000"],
    ]
    assert expected == generator.to_signals(inputs=[])


def test_reset_with_8_bits():
    generator = SignalNumberConverter(total_bits=8, frac_bits=0)
    expected = [["1", "0", "0" * 8, "0" * 8], ["0", "1", "0" * 8, "0" * 8]] * 2
    assert expected == generator.to_signals(inputs=[])


def test_0_5_converted_to_001():
    generator = SignalNumberConverter(total_bits=3, frac_bits=1)
    expected = [
        ["1", "0", "000", "000"],
        ["0", "1", "000", "000"],
        ["0", "0", "001", "001"],
        ["0", "1", "001", "001"],
        ["1", "0", "000", "000"],
        ["0", "1", "000", "000"],
    ]
    assert expected == generator.to_signals(inputs=[(0.5, 0.5)])


def test_UUU_to_NaN():
    converter = SignalNumberConverter(total_bits=3, frac_bits=1)
    assert math.isnan(converter.to_numbers(["UUU"])[0])


def test_101_to_2():
    converter = SignalNumberConverter(total_bits=3, frac_bits=1)
    assert [-1.5] == converter.to_numbers(["101"])
