from functools import partial
from unittest import TestCase

from elasticai.creator.vhdl.number_representations import (
    ClippedFixedPoint,
    FixedPoint,
    FixedPointFactory,
    ToLogicEncoder,
    fixed_point_params_from_factory,
    float_values_to_fixed_point,
    infer_total_and_frac_bits,
)


class FixedPointTest(TestCase):
    def test_zero(self):
        fp_value = FixedPoint(0, total_bits=1, frac_bits=0)
        self.assertEqual(0, int(fp_value))

    def test_1_with_8_total_bits_3_frac_bits(self):
        fp_value = FixedPoint(1, total_bits=8, frac_bits=3)
        self.assertEqual(8, int(fp_value))

    def test_minus_1_with_8_total_bits_3_frac_bits(self):
        fp_value = FixedPoint(-1, total_bits=8, frac_bits=3)
        self.assertEqual(248, int(fp_value))

    def test_minus_3_21_with_16_total_bits_12_frac_bits(self):
        fp_value = FixedPoint(-3.21, total_bits=16, frac_bits=12)
        self.assertEqual(52388, int(fp_value))

    def test_initialize_with_value_not_in_range(self):
        with self.assertRaises(ValueError):
            _ = FixedPoint(100, total_bits=4, frac_bits=2)

    def test_fixed_point_to_float_5_36(self):
        fp_value = FixedPoint(5.36, total_bits=12, frac_bits=6)
        self.assertAlmostEqual(5.36, float(fp_value), places=2)

    def test_fixed_point_to_float_minus_5_36(self):
        fp_value = FixedPoint(-5.36, total_bits=16, frac_bits=12)
        self.assertAlmostEqual(-5.36, float(fp_value), places=2)

    def test_to_hex_zero_with_one_bits(self):
        fp_value = FixedPoint(0, total_bits=1, frac_bits=0)
        self.assertEqual("0", fp_value.to_hex())

    def test_to_hex_zero_with_six_bits(self):
        fp_value = FixedPoint(0, total_bits=6, frac_bits=0)
        self.assertEqual("00", fp_value.to_hex())

    def test_to_hex_zero_with_sixteen_bits(self):
        fp_value = FixedPoint(0, total_bits=16, frac_bits=0)
        self.assertEqual("0000", fp_value.to_hex())

    def test_to_hex_minus_one_with_sixteen_bits(self):
        fp_value = FixedPoint(-1, total_bits=16, frac_bits=0)
        self.assertEqual("ffff", fp_value.to_hex())

    def test_to_hex_minus_three_with_three_bits(self):
        fp_value = FixedPoint(-3, total_bits=3, frac_bits=0)
        self.assertEqual("5", fp_value.to_hex())

    def test_to_hex_minus_254_with_sixteen_bits(self):
        fp_value = FixedPoint(-254, total_bits=16, frac_bits=0)
        self.assertEqual("ff02", fp_value.to_hex())

    def test_to_hex_minus_19_5_with_16_bits(self):
        fp_value = FixedPoint(-19.5, total_bits=16, frac_bits=8)
        self.assertEqual("ec80", fp_value.to_hex())

    def test_to_bin_zero_with_one_bits(self):
        fp_value = FixedPoint(0, total_bits=1, frac_bits=0)
        self.assertEqual("0", fp_value.to_bin())

    def test_to_bin_zero_with_three_bits(self):
        fp_value = FixedPoint(0, total_bits=3, frac_bits=0)
        self.assertEqual("000", fp_value.to_bin())

    def test_to_bin_five_with_four_bits(self):
        fp_value = FixedPoint(5, total_bits=4, frac_bits=0)
        self.assertEqual("0101", fp_value.to_bin())

    def test_to_bin_minus_one_with_two_bits(self):
        fp_value = FixedPoint(-1, total_bits=2, frac_bits=0)
        self.assertEqual("11", fp_value.to_bin())

    def test_to_bin_minus_two_with_two_bits(self):
        fp_value = FixedPoint(-2, total_bits=2, frac_bits=0)
        self.assertEqual("10", fp_value.to_bin())

    def test_to_bin_minus_256_with_sixteen_bits(self):
        fp_value = FixedPoint(-256, total_bits=16, frac_bits=0)
        self.assertEqual("1111111100000000", fp_value.to_bin())

    def test_to_bin_minus_254_with_sixteen_bits(self):
        fp_value = FixedPoint(-254, total_bits=16, frac_bits=0)
        self.assertEqual("1111111100000010", fp_value.to_bin())

    def test_to_bin_minus_19_5_with_16_bits(self):
        fp_value = FixedPoint(-19.5, total_bits=16, frac_bits=8)
        self.assertEqual("1110110010000000", fp_value.to_bin())

    def test_from_unsigned_int(self):
        fp_value = FixedPoint.from_unsigned_int(52388, total_bits=16, frac_bits=12)
        self.assertAlmostEqual(-3.21, float(fp_value), places=2)

    def test_from_signed_int(self):
        fp_value = FixedPoint.from_signed_int(-13148, total_bits=16, frac_bits=12)
        self.assertAlmostEqual(-3.21, float(fp_value), places=2)

    def test_repr(self):
        fp_value = FixedPoint(value=3.2, total_bits=8, frac_bits=4)
        target_repr = "FixedPoint(value=3.2, total_bits=8, frac_bits=4)"
        self.assertEqual(target_repr, repr(fp_value))

    def test_str(self):
        fp_value = FixedPoint(5, total_bits=8, frac_bits=4)
        target_str = "80"
        self.assertEqual(target_str, str(fp_value))

    def test_bin_iter(self):
        fp_value = FixedPoint(-19.5, total_bits=16, frac_bits=8)
        actual_bin = list(fp_value.bin_iter())[::-1]
        expected_bin = [1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(expected_bin, actual_bin)

    def test_neg_0(self):
        fp_value = FixedPoint(0, total_bits=8, frac_bits=2)
        self.assertEqual(0, float(-fp_value))

    def test_neg_minus_5(self):
        fp_value = FixedPoint(-5, total_bits=8, frac_bits=2)
        self.assertEqual(5, float(-fp_value))

    def test_neg_9(self):
        fp_value = FixedPoint(9, total_bits=8, frac_bits=2)
        self.assertEqual(-9, float(-fp_value))

    def test_invert_6(self):
        fp_value = FixedPoint(6, total_bits=8, frac_bits=2)
        self.assertEqual(231, int(~fp_value))

    def test_invert_minus_15(self):
        fp_value = FixedPoint(-15, total_bits=8, frac_bits=2)
        self.assertEqual(59, int(~fp_value))

    def test_abs_minus_5(self):
        fp_value = FixedPoint(-5, total_bits=8, frac_bits=2)
        self.assertEqual(5, abs(fp_value))

    def test_abs_5(self):
        fp_value = FixedPoint(5, total_bits=8, frac_bits=2)
        self.assertEqual(5, abs(fp_value))

    def test_eq(self):
        to_fp = partial(FixedPoint, total_bits=8, frac_bits=4)
        self.assertEqual(to_fp(5.9), to_fp(5.9))

    def test_ne(self):
        to_fp = partial(FixedPoint, total_bits=8, frac_bits=4)
        self.assertNotEqual(to_fp(4.9), to_fp(4.2))

    def test_lt(self):
        to_fp = partial(FixedPoint, total_bits=8, frac_bits=4)
        self.assertLess(to_fp(4.2), to_fp(4.9))

    def test_le(self):
        to_fp = partial(FixedPoint, total_bits=8, frac_bits=4)
        self.assertLessEqual(to_fp(4.2), to_fp(4.9))
        self.assertLessEqual(to_fp(4.2), to_fp(4.2))

    def test_gt(self):
        to_fp = partial(FixedPoint, total_bits=8, frac_bits=4)
        self.assertGreater(to_fp(4.9), to_fp(4.2))

    def test_ge(self):
        to_fp = partial(FixedPoint, total_bits=8, frac_bits=4)
        self.assertGreaterEqual(to_fp(4.9), to_fp(4.2))
        self.assertGreaterEqual(to_fp(4.9), to_fp(4.9))

    def test_add(self):
        fp1 = FixedPoint(-2.5, total_bits=8, frac_bits=4)
        fp2 = FixedPoint(5, total_bits=8, frac_bits=4)
        result = FixedPoint(2.5, total_bits=8, frac_bits=4)
        self.assertEqual(result, fp1 + fp2)

    def test_add_overflow(self):
        fp1 = FixedPoint(4, total_bits=8, frac_bits=4)
        fp2 = FixedPoint(5, total_bits=8, frac_bits=4)
        result = FixedPoint(-7, total_bits=8, frac_bits=4)
        self.assertEqual(result, fp1 + fp2)

    def test_add_incompatible(self):
        fp1 = FixedPoint(3, total_bits=8, frac_bits=4)
        fp2 = FixedPoint(2.5, total_bits=8, frac_bits=2)
        with self.assertRaises(ValueError):
            _ = fp1 + fp2

    def test_sub(self):
        fp1 = FixedPoint(3, total_bits=8, frac_bits=4)
        fp2 = FixedPoint(4.5, total_bits=8, frac_bits=4)
        result = FixedPoint(-1.5, total_bits=8, frac_bits=4)
        self.assertEqual(result, fp1 - fp2)

    def test_sub_overflow(self):
        fp1 = FixedPoint(-4, total_bits=8, frac_bits=4)
        fp2 = FixedPoint(5, total_bits=8, frac_bits=4)
        result = FixedPoint(7, total_bits=8, frac_bits=4)
        self.assertEqual(result, fp1 - fp2)

    def test_sub_incompatible(self):
        fp1 = FixedPoint(3, total_bits=8, frac_bits=4)
        fp2 = FixedPoint(2.5, total_bits=8, frac_bits=2)
        with self.assertRaises(ValueError):
            _ = fp1 - fp2

    def test_and(self):
        fp1 = FixedPoint(5, total_bits=4, frac_bits=0)
        fp2 = FixedPoint(6, total_bits=4, frac_bits=0)
        result = FixedPoint(4, total_bits=4, frac_bits=0)
        self.assertEqual(result, fp1 & fp2)

    def test_or(self):
        fp1 = FixedPoint(5, total_bits=4, frac_bits=0)
        fp2 = FixedPoint(6, total_bits=4, frac_bits=0)
        result = FixedPoint(7, total_bits=4, frac_bits=0)
        self.assertEqual(result, fp1 | fp2)

    def test_xor(self):
        fp1 = FixedPoint(5, total_bits=4, frac_bits=0)
        fp2 = FixedPoint(6, total_bits=4, frac_bits=0)
        result = FixedPoint(3, total_bits=4, frac_bits=0)
        self.assertEqual(result, fp1 ^ fp2)

    def test_to_signed_int_positive_value(self):
        fp = FixedPoint(5, total_bits=8, frac_bits=4)
        self.assertEqual(80, fp.to_signed_int())

    def test_to_signed_int_negative_value(self):
        fp = FixedPoint(-5, total_bits=8, frac_bits=4)
        self.assertEqual(-80, fp.to_signed_int())


class ClippedFixedPointTest(TestCase):
    def test_conversion_value_in_bounds(self) -> None:
        fp = ClippedFixedPoint(5.251, total_bits=8, frac_bits=4)
        self.assertEqual(84, int(fp))
        self.assertEqual(5.25, float(fp))

    def test_conversion_value_out_of_lower_bound(self) -> None:
        fp = ClippedFixedPoint(-9.25, total_bits=8, frac_bits=4)
        self.assertEqual(128, int(fp))
        self.assertEqual(-8, float(fp))

    def test_conversion_value_out_of_upper_bound(self) -> None:
        fp = ClippedFixedPoint(8, total_bits=8, frac_bits=4)
        self.assertEqual(127, int(fp))
        self.assertEqual(7.9375, float(fp))

    def test_repr_value_out_of_bounds(self) -> None:
        fp = ClippedFixedPoint(10, total_bits=8, frac_bits=4)
        self.assertEqual(
            "ClippedFixedPoint(value=7.9375, total_bits=8, frac_bits=4)", repr(fp)
        )

    def test_repr_value_in_bounds(self) -> None:
        fp = ClippedFixedPoint(5.21, total_bits=8, frac_bits=4)
        self.assertEqual(
            "ClippedFixedPoint(value=5.21, total_bits=8, frac_bits=4)", repr(fp)
        )

    def test_from_unsigned_int_value_in_bounds(self) -> None:
        fp = ClippedFixedPoint.from_unsigned_int(62, total_bits=8, frac_bits=4)
        self.assertEqual(62, int(fp))
        self.assertEqual(3.875, float(fp))

    def test_from_unsigned_int_value_out_of_bounds(self) -> None:
        fp = ClippedFixedPoint.from_unsigned_int(830, total_bits=8, frac_bits=4)
        self.assertEqual(62, int(fp))
        self.assertEqual(3.875, float(fp))

    def test_from_signed_int_value_in_bounds(self) -> None:
        fp = ClippedFixedPoint.from_signed_int(-100, total_bits=8, frac_bits=4)
        self.assertEqual(156, int(fp))
        self.assertEqual(-6.25, float(fp))

    def test_from_signed_int_value_out_of_bounds(self) -> None:
        fp = ClippedFixedPoint.from_signed_int(-255, total_bits=8, frac_bits=4)
        self.assertEqual(128, int(fp))
        self.assertEqual(-8, float(fp))

    def test_get_factory(self) -> None:
        factory = ClippedFixedPoint.get_factory(total_bits=8, frac_bits=4)
        fp = factory(1)
        self.assertEqual(ClippedFixedPoint, type(fp))
        self.assertEqual(8, fp.total_bits)
        self.assertEqual(4, fp.frac_bits)
        self.assertEqual(1, float(fp))


class InferTotalAndFracBits(TestCase):
    def test_infer_empty_list(self):
        with self.assertRaises(ValueError):
            _, _ = infer_total_and_frac_bits([])

    def test_infer_mixed_total_bits(self):
        with self.assertRaises(ValueError):
            _, _ = infer_total_and_frac_bits(
                [
                    FixedPoint(0, total_bits=8, frac_bits=4),
                    FixedPoint(0, total_bits=8, frac_bits=4),
                    FixedPoint(0, total_bits=12, frac_bits=4),
                    FixedPoint(0, total_bits=8, frac_bits=4),
                ]
            )

    def test_infer_mixed_frac_bits(self):
        with self.assertRaises(ValueError):
            _, _ = infer_total_and_frac_bits(
                [
                    FixedPoint(0, total_bits=8, frac_bits=4),
                    FixedPoint(0, total_bits=8, frac_bits=4),
                    FixedPoint(0, total_bits=8, frac_bits=5),
                    FixedPoint(0, total_bits=8, frac_bits=4),
                ]
            )

    def test_infer_mixed_total_and_frac_bits(self):
        with self.assertRaises(ValueError):
            _, _ = infer_total_and_frac_bits(
                [
                    FixedPoint(0, total_bits=8, frac_bits=4),
                    FixedPoint(0, total_bits=8, frac_bits=4),
                    FixedPoint(0, total_bits=8, frac_bits=5),
                    FixedPoint(0, total_bits=12, frac_bits=4),
                ]
            )

    def test_infer_valid_list(self):
        total_bits, frac_bits = infer_total_and_frac_bits(
            [
                FixedPoint(0, total_bits=8, frac_bits=4),
                FixedPoint(0, total_bits=8, frac_bits=4),
            ]
        )
        self.assertEqual(8, total_bits)
        self.assertEqual(4, frac_bits)

    def test_infer_multiple_valid_lists(self):
        values = [
            FixedPoint(0, total_bits=8, frac_bits=4),
            FixedPoint(0, total_bits=8, frac_bits=4),
        ]
        total_bits, frac_bits = infer_total_and_frac_bits(values, values, values)
        self.assertEqual(8, total_bits)
        self.assertEqual(4, frac_bits)

    def test_infer_multiple_invalid_lists(self):
        values1 = [
            FixedPoint(0, total_bits=8, frac_bits=4),
            FixedPoint(0, total_bits=8, frac_bits=4),
        ]
        values2 = [
            FixedPoint(0, total_bits=12, frac_bits=4),
            FixedPoint(0, total_bits=8, frac_bits=4),
        ]
        with self.assertRaises(ValueError):
            _, _ = infer_total_and_frac_bits(values1, values2)

    def test_infer_multiple_lists_with_empty_list(self):
        values = [
            FixedPoint(0, total_bits=8, frac_bits=4),
            FixedPoint(0, total_bits=8, frac_bits=4),
        ]
        with self.assertRaises(ValueError):
            _, _ = infer_total_and_frac_bits(values, [])


def FixedPointParamsFromFactoryTest(TestCase):
    def test_fixed_point_params_from_factory_8total_4frac_bits(self) -> None:
        target_total_bits, taret_frac_bits = 8, 4
        factory = FixedPoint.get_factory(taret_frac_bits, target_total_bits)
        total_bits, frac_bits = fixed_point_params_from_factory(factory)
        self.assertEqual(target_total_bits, total_bits)
        self.assertEqual(taret_frac_bits, frac_bits)

    def test_fixed_point_params_from_factory_clipped_8total_4frac_bits(self) -> None:
        target_total_bits, taret_frac_bits = 8, 4
        factory = ClippedFixedPoint.get_factory(taret_frac_bits, target_total_bits)
        total_bits, frac_bits = fixed_point_params_from_factory(factory)
        self.assertEqual(target_total_bits, total_bits)
        self.assertEqual(taret_frac_bits, frac_bits)

    def test_fixed_point_params_from_factory_8total_0frac_bits(self) -> None:
        target_total_bits, taret_frac_bits = 8, 0
        factory = FixedPoint.get_factory(taret_frac_bits, target_total_bits)
        total_bits, frac_bits = fixed_point_params_from_factory(factory)
        self.assertEqual(target_total_bits, total_bits)
        self.assertEqual(taret_frac_bits, frac_bits)


class FloatValuesToFixedPointTest(TestCase):
    def test_empty_list(self):
        actual = float_values_to_fixed_point([], total_bits=8, frac_bits=4)
        self.assertListEqual([], actual)

    def test_full_list(self):
        actual = float_values_to_fixed_point(
            values=[1, 2, 3],
            total_bits=8,
            frac_bits=4,
        )
        target = [FixedPoint(value, total_bits=8, frac_bits=4) for value in range(1, 4)]
        self.assertListEqual(target, actual)


class IntValuesToFixedPointTest(TestCase):
    def test_empty_list(self):
        actual = float_values_to_fixed_point([], total_bits=8, frac_bits=0)
        self.assertListEqual([], actual)

    def test_full_list(self):
        actual = float_values_to_fixed_point(
            values=[1, 2, 3],
            total_bits=8,
            frac_bits=0,
        )
        target = [FixedPoint(value, total_bits=8, frac_bits=4) for value in range(1, 4)]
        self.assertListEqual(target, actual)


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
        encoder.register_symbol(-1)
        self.assertEqual(0, encoder[-1])

    def test_binarization_minus_1_to_0_and_1to1(self):
        encoder = ToLogicEncoder()
        encoder.register_symbol(-1)
        encoder.register_symbol(1)
        self.assertEqual(1, encoder[1])

    def test_encoder_is_monotonous(self):
        encoder = ToLogicEncoder()
        encoder.register_symbol(1)
        encoder.register_symbol(-1)
        self.assertEqual(1, encoder[1])

    def test_encoder_to_bit_vector(self):
        encoder = ToLogicEncoder()
        encoder.register_symbol(1)
        encoder.register_symbol(-1)
        bits = encoder(-1)
        self.assertEqual("0", bits)

    def test_ternarization_minus1_to_00(self):
        encoder = ToLogicEncoder()
        encoder.register_symbol(-1)
        encoder.register_symbol(0)
        encoder.register_symbol(1)
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

    def test_registering_numbers_in_batch(self):
        one_by_one = ToLogicEncoder()
        by_batch = ToLogicEncoder()
        batch = [1, 0, -1]
        for number in batch:
            one_by_one.register_symbol(number)
        by_batch.register_symbols(batch)
        self.assertTrue(
            by_batch == one_by_one,
            "expected: {}, actual: {}".format(one_by_one, by_batch),
        )
