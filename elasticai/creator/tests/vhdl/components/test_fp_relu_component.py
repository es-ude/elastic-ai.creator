import unittest

from elasticai.creator.vhdl.components.fp_relu_component import FPReLUComponent
from elasticai.creator.vhdl.number_representations import FixedPoint


class FPReLUComponentTest(unittest.TestCase):
    def test_relu_correct_number_of_lines(self) -> None:
        to_fp = FixedPoint.get_factory(total_bits=8, frac_bits=4)

        component = FPReLUComponent(
            fixed_point_factory=to_fp,
        )

        self.assertEqual(len(list(component())), 66)
