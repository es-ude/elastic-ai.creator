import unittest

from elasticai.creator.vhdl.code_files.fp_linear_1d_component import FPLinear1dFile
from elasticai.creator.vhdl.number_representations import FixedPoint


class FPLinear1dComponentTest(unittest.TestCase):
    def test_linear_1d_correct_number_of_lines(self) -> None:
        to_fp = FixedPoint.get_factory(total_bits=8, frac_bits=4)

        component = FPLinear1dFile(
            layer_id="ll1",
            in_feature_num=3,
            out_feature_num=2,
            fixed_point_factory=to_fp,
            work_library_name="work",
        )

        self.assertEqual(len(list(component.code())), 224)
