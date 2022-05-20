from unittest import TestCase

from elasticai.creator.vhdl.number_representations import (
    FloatToSignedFixedPointConverter, ToLogicEncoder, hex_representation,
    two_complements_representation)


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


class HexRepresentation(TestCase):
    def test_one(self):
        actual = hex_representation(1, 16)
        expected = 'x"0001"'
        self.assertEqual(expected, actual)

    def test_minus_one(self):
        actual = hex_representation(-1, 16)
        expected = 'x"ffff"'
        self.assertEqual(expected, actual)

    def test_two(self):
        actual = hex_representation(2, 16)
        expected = 'x"0002"'
        self.assertEqual(expected, actual)

    def test_minus_two(self):
        actual = hex_representation(-2, 16)
        expected = 'x"fffe"'
        self.assertEqual(expected, actual)

    def test_minus_four_four_bit(self):
        actual = hex_representation(-4, 4)
        expected = 'x"c"'
        self.assertEqual(expected, actual)

    def test_minus_three_three_bit(self):
        actual = hex_representation(-3, 3)
        expected = 'x"5"'
        self.assertEqual(expected, actual)

    def test_minus_256_16_bit(self):
        actual = hex_representation(-256, 16)
        expected = 'x"ff00"'
        self.assertEqual(expected, actual)

    def test_minus_254_16_bit(self):
        actual = hex_representation(-254, 16)
        expected = 'x"ff02"'
        self.assertEqual(expected, actual)


class NumberEncoderTest(TestCase):
    """
    Test Cases:
      - build new encoder from existing encoder ensuring compatibility of enumerations
        Use case scenario: Connecting the outputs of layer h_1 to the inputs of layer h_2, while we can consider the in-
         and output as enumerations, ie. we don't care about the actual numeric values. However to still allow for
         max pooling operations we might want to ensure that the encoding is monotonous.
    """

    def test_binarization_minus_one_is_zero(self):
        encoder = ToLogicEncoder()
        encoder.add_numeric(-1)
        self.assertEqual(0, encoder[-1])

    def test_binarization_minus_1_to_0_and_1to1(self):
        encoder = ToLogicEncoder()
        encoder.add_numeric(-1)
        encoder.add_numeric(1)
        self.assertEqual(1, encoder[1])

    def test_encoder_is_monotonous(self):
        encoder = ToLogicEncoder()
        encoder.add_numeric(1)
        encoder.add_numeric(-1)
        self.assertEqual(1, encoder[1])

    def test_encoder_to_bit_vector(self):
        encoder = ToLogicEncoder()
        encoder.add_numeric(1)
        encoder.add_numeric(-1)
        bits = encoder(-1)
        self.assertEqual("0", bits)

    def test_ternarization_minus1_to_00(self):
        encoder = ToLogicEncoder()
        encoder.add_numeric(-1)
        encoder.add_numeric(0)
        encoder.add_numeric(1)
        test_parameters = (
            (0, -1),
            (1, 0),
            (2, 1),
        )
        for parameter in test_parameters:
            with self.subTest(parameter):
                expected = parameter[0]
                actual = encoder[parameter[1]]
                self.assertEqual(expected, actual)
