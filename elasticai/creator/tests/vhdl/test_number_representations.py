from unittest import TestCase
from elasticai.creator.vhdl.number_representations import (
    FloatToSignedFixedPointConverter,
    two_complements_representation,
)


class FixedPointConverterTest(TestCase):
    def test_get_zero(self):
        f = FloatToSignedFixedPointConverter(bits_used_for_fraction=0)
        self.assertEqual(0, f(0))

    def test_get_one_with_2bits_for_fraction(self):
        f = FloatToSignedFixedPointConverter(bits_used_for_fraction=2)
        self.assertEqual(1 << 2, f(1))

    def test_get_one_with_3bits_for_fraction(self):
        f = FloatToSignedFixedPointConverter(bits_used_for_fraction=3)
        self.assertEqual(1 << 3, f(1))

    def test_raise_error_if_not_convertible(self):
        f = FloatToSignedFixedPointConverter(bits_used_for_fraction=0)
        try:
            f(0.5)
            self.fail()
        except ValueError as e:
            self.assertEqual(
                "0.5 not convertible to fixed point number using 0 bits for fractional part",
                str(e),
            )


class BinaryTwoComplementRepresentation(TestCase):
    def test_one(self):
        actual = two_complements_representation(1, 1)
        expected = "1"
        self.assertEqual(expected, actual)

    def test_minus_one(self):
        actual = two_complements_representation(-1, 2)
        expected = "11"
        self.assertEqual(expected, actual)

    def test_minus_two(self):
        actual = two_complements_representation(-2, 2)
        expected = "10"
        self.assertEqual(expected, actual)

    def test_two(self):
        actual = two_complements_representation(2, 3)
        expected = "010"
        self.assertEqual(expected, actual)

    def test_minus_four(self):
        actual = two_complements_representation(-4, 3)
        expected = "100"
        self.assertEqual(expected, actual)

    def test_minus_three_three_bit(self):
        actual = two_complements_representation(-3, 3)
        expected = "101"
        self.assertEqual(expected, actual)

    def test_minus_256_16_bit(self):
        actual = two_complements_representation(-256, 16)
        expected = "1111111100000000"
        self.assertEqual(expected, actual)

    def test_minus_254_16_bit(self):
        actual = two_complements_representation(-254, 16)
        expected = "1111111100000010"
        self.assertEqual(expected, actual)
