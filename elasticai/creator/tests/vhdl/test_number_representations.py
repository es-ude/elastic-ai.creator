from unittest import TestCase
from elasticai.creator.vhdl.number_representations import FixedPointConverter


class FixedPointTest(TestCase):
    def test_get_zero(self):
        f = FixedPointConverter(bits_used_for_fraction=0)
        self.assertEqual(0, f.from_float(0))

    def test_get_one_with_2bits_for_fraction(self):
        f = FixedPointConverter(bits_used_for_fraction=2)
        self.assertEqual(1 << 2, f.from_float(1))

    def test_get_one_with_3bits_for_fraction(self):
        f = FixedPointConverter(bits_used_for_fraction=3)
        self.assertEqual(1 << 3, f.from_float(1))

    def test_raise_error_if_not_convertible(self):
        f = FixedPointConverter(bits_used_for_fraction=0)
        try:
            f.from_float(0.5)
            self.fail()
        except ValueError as e:
            self.assertEqual(
                "0.5 not convertible to fixed point number using 0 bits for fractional part",
                str(e),
            )
