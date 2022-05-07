import unittest
from elasticai.creator.vhdl.generator.rom import (
    pad_with_zeros,
)


class HexArrayTests(unittest.TestCase):
    def test_fill_with_zeros_up_to_power_of_two(self):
        hex_list = ["ff", "a0", "c1"]
        expected_list = ["ff", "a0", "c1", "00"]
        actual_list = pad_with_zeros(hex_list)
        self.assertEqual(expected_list, actual_list)

    def test_input_with_one_element_still_appends_one_zero_element(self):
        hex_list = ["ff"]
        expected_list = ["ff", "00"]
        actual_list = pad_with_zeros(hex_list)
        self.assertEqual(expected_list, actual_list)
