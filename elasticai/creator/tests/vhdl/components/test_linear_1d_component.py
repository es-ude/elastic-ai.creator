import unittest

from elasticai.creator.vhdl.components import Linear1dComponent
from elasticai.creator.vhdl.number_representations import FixedPoint


class Linear1dComponentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.component = Linear1dComponent(
            in_features=20,
            out_features=1,
            fixed_point_factory=FixedPoint.get_factory(total_bits=16, frac_bits=8),
            work_library_name="work",
        )

    def test_derives_correct_data_width(self) -> None:
        self.assertEqual(self.component.data_width, 16)

    def test_calculates_correct_addr_width(self) -> None:
        self.assertEqual(self.component.addr_width, 5)

    def test_out_features_larger_1_raises_not_implemented_error(self) -> None:
        with self.assertRaises(NotImplementedError):
            _ = Linear1dComponent(
                in_features=3,
                out_features=2,
                fixed_point_factory=FixedPoint.get_factory(total_bits=8, frac_bits=4),
            )
