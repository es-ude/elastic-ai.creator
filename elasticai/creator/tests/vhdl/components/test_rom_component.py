import unittest

from elasticai.creator.vhdl.components import RomComponent
from elasticai.creator.vhdl.number_representations import FixedPoint


class RomComponentTest(unittest.TestCase):
    def setUp(self) -> None:
        fp = FixedPoint.get_factory(total_bits=16, frac_bits=8)
        self.rom = RomComponent(
            rom_name="test_rom",
            values=[fp(i) for i in range(20)],
            resource_option="auto",
        )

    def test_data_width_correct_derived(self) -> None:
        self.assertEqual(self.rom.data_width, 16)

    def test_addr_width_correct_calculated(self) -> None:
        self.assertEqual(self.rom.addr_width, 5)

    def test_correct_number_of_values(self) -> None:
        self.assertEqual(len(self.rom.hex_values), 32)

    def test_values_correct_padded(self) -> None:
        self.assertEqual(['x"0000"'] * 12, self.rom.hex_values[20:])
