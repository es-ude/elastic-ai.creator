import unittest

from elasticai.creator.vhdl.code_files.lstm_component import LSTMFile
from elasticai.creator.vhdl.number_representations import FixedPoint


class LSTMComponentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lstm = LSTMFile(
            input_size=5,
            hidden_size=3,
            fixed_point_factory=FixedPoint.get_factory(total_bits=8, frac_bits=4),
            layer_id="0",
            work_library_name="xil_defaultlib",
        )

    def test_fixed_point_params_correct_derived(self):
        self.assertEqual(8, self.lstm.data_width)
        self.assertEqual(4, self.lstm.frac_width)

    def test_x_h_addr_width_correct_set(self):
        self.assertEqual(3, self.lstm.x_h_addr_width)

    def test_hidden_addr_width_correct_set(self):
        self.assertEqual(2, self.lstm.hidden_addr_width)

    def test_w_addr_width_correct_set(self):
        self.assertEqual(5, self.lstm.w_addr_width)
