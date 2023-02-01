import unittest

from elasticai.creator.vhdl.code_files.fp_hard_tanh_component import FPHardTanhComponent
from elasticai.creator.vhdl.number_representations import FixedPoint


class FPHardSigmoidComponentTest(unittest.TestCase):
    def test_hard_sigmoid_correct_number_of_lines(self) -> None:
        to_fp = FixedPoint.get_factory(total_bits=8, frac_bits=4)

        component = FPHardTanhComponent(
            min_val=to_fp(-1),
            max_val=to_fp(1),
            fixed_point_factory=to_fp,
            layer_id="0",
        )

        self.assertEqual(len(list(component.code())), 52)
