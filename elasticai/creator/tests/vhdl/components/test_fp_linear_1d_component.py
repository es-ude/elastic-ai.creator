import unittest

from elasticai.creator.vhdl.components.fp_linear_1d_component import FPLinear1dComponent
from elasticai.creator.vhdl.number_representations import FixedPoint


class FPLinear1dComponentTest(unittest.TestCase):
    def test_linear_1d_correct_number_of_lines(self) -> None:
        to_fp = FixedPoint.get_factory(total_bits=8, frac_bits=4)

        component = FPLinear1dComponent(
            layer_name="ll1",
            in_features=3,
            out_features=2,
            fixed_point_factory=to_fp,
            work_library_name="work",
        )

        self.assertEqual(len(list(component())), 204)
