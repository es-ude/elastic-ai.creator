import unittest

from elasticai.creator.vhdl.components.network_component import (
    SignalsForComponentWithBuffer,
    SignalsForBufferlessComponent,
)


class ComponentSignalDeclarationsTest(unittest.TestCase):
    def constructor(self, name, data_width, *args):
        return SignalsForBufferlessComponent(name, data_width=data_width)

    def test_clock_signal_is_generated(self):
        signals = self.constructor("other_name", 2)
        self.assertTrue(
            "signal other_name_clock : std_logic := '0';" in list(signals.code())
        )

    def test_component_name_is_generated(self):
        signals = self.constructor("other_name", 2)
        self.assertTrue(
            "signal other_name_enable : std_logic := '0';" in list(signals.code())
        )

    def check_signals_contain_logic_vector(self, signals, suffix, width):
        expected = (
            f"signal {signals.name}_{suffix} : std_logic_vector({width - 1} downto 0);"
        )
        self.assertTrue(
            expected in list(signals.code()),
            msg=f"expected: {expected}\nfound: {list(signals.code())}",
        )

    def test_generates_correct_number_of_lines(self):
        signals = self.constructor("other_name", 2)
        self.assertEqual(4, len(tuple(signals.code())))


class BufferedComponentSignalsTest(ComponentSignalDeclarationsTest):
    def constructor(self, name, data_width, *args):
        x_address_width = 1
        y_address_width = 1
        if len(args) == 2:
            x_address_width = args[0]
            y_address_width = args[1]
        return SignalsForComponentWithBuffer(
            name, data_width, x_address_width, y_address_width
        )

    def test_5_downto_0_generated_for_x_addr_width_6(self):
        signals = self.constructor("other_name", 4, 6, 2)
        self.check_signals_contain_logic_vector(signals, "x_address", 6)

    def test_8_downto_0_generated_for_y_addr_width_9(self):
        signals = self.constructor("other_name", 4, 6, 9)
        self.check_signals_contain_logic_vector(signals, "y_address", 9)

    def test_generates_correct_number_of_lines(self):
        signals = self.constructor("other_name", 2, 2, 4)
        self.assertEqual(7, len(tuple(signals.code())))


if __name__ == "__main__":
    unittest.main()
