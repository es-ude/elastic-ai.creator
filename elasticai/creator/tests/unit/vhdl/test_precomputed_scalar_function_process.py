import unittest
from functools import partial

from elasticai.creator.vhdl.number_representations import float_values_to_fixed_point
from elasticai.creator.vhdl.precomputed_scalar_function import (
    precomputed_scalar_function_process,
)


class PrecomputedScalarFunctionProcessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fp_list = partial(float_values_to_fixed_point, total_bits=16, frac_bits=8)

    def test_empty_x_list_y_list_one_element(self) -> None:
        self.assertEqual(
            list(precomputed_scalar_function_process(x=[], y=self.fp_list([1]))()),
            ['y <= "0000000100000000";'],
        )

    def test_empty_x_list_y_list_too_many_elements(self) -> None:
        with self.assertRaises(ValueError):
            list(precomputed_scalar_function_process(x=[], y=self.fp_list([1, 2]))())

    def test_empty_y_list(self) -> None:
        with self.assertRaises(ValueError):
            list(precomputed_scalar_function_process(x=[], y=[])())

    def test_x_list_lengths_not_suitable_for_y_list_lengths(self) -> None:
        with self.assertRaises(ValueError):
            list(
                precomputed_scalar_function_process(
                    x=self.fp_list([1, 2]), y=self.fp_list([1])
                )()
            )

    def test_x_list_with_only_one_element(self) -> None:
        expected_code = [
            "if int_x<256 then",
            '\ty <= "0000000100000000"; -- 256',
            "else",
            '\ty <= "0000001000000000"; -- 512',
            "end if;",
        ]
        self.assertEqual(
            expected_code,
            list(
                precomputed_scalar_function_process(
                    x=self.fp_list([1]), y=self.fp_list([1, 2])
                )()
            ),
        )

    def test_unsorted_x_list(self) -> None:
        expected_code = [
            "if int_x<256 then",
            '\ty <= "0000000100000000"; -- 256',
            "elsif int_x<512 then",
            '\ty <= "0000001000000000"; -- 512',
            "elsif int_x<768 then",
            '\ty <= "0000001100000000"; -- 768',
            "else",
            '\ty <= "0000010000000000"; -- 1024',
            "end if;",
        ]
        self.assertEqual(
            expected_code,
            list(
                precomputed_scalar_function_process(
                    x=self.fp_list([3, 1, 2]), y=self.fp_list([1, 2, 3, 4])
                )()
            ),
        )
