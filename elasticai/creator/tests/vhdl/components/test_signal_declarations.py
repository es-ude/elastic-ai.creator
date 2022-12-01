import unittest

from elasticai.creator.vhdl.components.network_component import SignalsForComponentWithBuffer, \
    SignalsForBufferlessComponent


class ComponentSignalDeclarationsTest(unittest.TestCase):
    constructor = SignalsForBufferlessComponent

    def test_clock_signal_is_generated(self):
        signals = self.constructor("other_name", 2, 2, 2)
        self.assertTrue("signal other_name_clock : std_logic := '0';" in list(signals.code()))

    def test_component_name_is_generated(self):
        signals = self.constructor("other_name", 2, 2, 2)
        self.assertTrue("signal other_name_enable : std_logic := '0';" in list(signals.code()))

    def check_signals_contain_logic_vector(self, signals, suffix, width):
        self.assertTrue(f"signal {signals.name}_{suffix} : std_logic_vector({width - 1} downto 0);" in list(signals.code()))

    def test_3_downto_0_generated_for_input_data_width_4(self):
        signals = self.constructor("other_name", 2, 2, 4)
        self.check_signals_contain_logic_vector(signals, "input", 4)

    def test_3_downto_0_generated_for_output_data_width_4(self):
        signals = self.constructor("other_name", 2, 2, 4)
        self.check_signals_contain_logic_vector(signals, "output", 4)

    def test_generates_correct_number_of_lines(self):
        signals = self.constructor("other_name", 2, 2, 4)
        self.assertEqual(4, len(tuple(signals.code())))


class BufferedComponentSignalsTest(ComponentSignalDeclarationsTest):
    constructor = SignalsForComponentWithBuffer

    def test_5_downto_0_generated_for_x_addr_width_6(self):
        signals = self.constructor("other_name", 6, 2, 4)
        self.check_signals_contain_logic_vector(signals, "input_addr_width", 6)

    def test_8_downto_0_generated_for_y_addr_width_9(self):
        signals = self.constructor("other_name", 6, 9, 4)
        self.check_signals_contain_logic_vector(signals, "output_addr_width", 9)

    def test_generates_correct_number_of_lines(self):
        signals = self.constructor("other_name", 2, 2, 4)
        self.assertEqual(7, len(tuple(signals.code())))


if __name__ == '__main__':
    unittest.main()
