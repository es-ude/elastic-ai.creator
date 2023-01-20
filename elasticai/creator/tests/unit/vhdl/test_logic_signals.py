import unittest

from elasticai.creator.vhdl.signals import (
    LogicInSignal,
    LogicInVectorSignal,
    LogicOutSignal,
    LogicOutVectorSignal,
)


class LogicSignalTestCase(unittest.TestCase):
    def test_logic_in_signal_definition_is_p_x(self):
        in_signal = LogicInSignal(basename="x").with_prefix(prefix="p")
        self.assertEqual("signal p_x : std_logic;", in_signal.definition())

    def test_logic_in_signal_definition_is_y(self):
        in_signal = LogicInSignal(basename="y")
        self.assertEqual("signal y : std_logic;", in_signal.definition())

    def test_logic_vector_in_signal_definition_is_x_0_downto_0(self):
        definition = LogicInVectorSignal(basename="x", width=1).definition()
        self.assertEqual("signal x : std_logic_vector(0 downto 0);", definition)

    def test_logic_vector_in_signal_definition_is_y_3_downto_0_with_default_and_prefix(
        self,
    ):
        definition = (
            LogicInVectorSignal(basename="x", width=4, default_value="(other => '0')")
            .with_prefix("p")
            .definition()
        )
        self.assertEqual(
            "signal p_x : std_logic_vector(3 downto 0) := (other => '0');", definition
        )

    def test_reversing_InSignal_returns_OutSignal_with_matching_defintion(self):
        in_signal = LogicInSignal(basename="x").with_prefix("p")
        out_signal = in_signal.reverse()
        self.assertIsInstance(out_signal, LogicOutSignal)
        self.assertEqual(in_signal.definition(), out_signal.definition())

    def test_reversing_out_returns_in_with_matching_definition(self):
        out_signal = LogicOutSignal(basename="x").with_prefix("p")
        in_signal = out_signal.reverse()
        self.assertIsInstance(in_signal, LogicInSignal)
        self.assertEqual(out_signal.definition(), in_signal.definition())

    def test_connecting_matching_logic_signals_returns_code(self):
        in_signal = LogicInSignal(basename="x")
        out_signal = LogicOutSignal(basename="x").with_prefix("p")
        in_signal.connect(out_signal)
        self.assertEqual(["x <= p_x;"], list(in_signal.code()))

    def test_connecting_non_matching_signals_returns_empty_iterable(self):
        in_signal = LogicInSignal(basename="x")
        out_signal = LogicOutSignal(basename="y").with_prefix("p")
        in_signal.connect(out_signal)
        self.assertEqual([], list(in_signal.code()))

    def test_signals_of_different_name_dont_connect(self):
        in_signal = LogicInSignal(basename="x")
        out_signal = LogicOutSignal(basename="y")
        in_signal.connect(out_signal)
        self.assertTrue(in_signal.is_missing_inputs())

    def test_connecting_matching_vector_signals_returns_code(self):
        in_signal = LogicInVectorSignal(basename="x", width=2)
        out_signal = LogicOutVectorSignal(basename="x", width=2).with_prefix("p")
        in_signal.connect(out_signal)
        self.assertEqual(["x <= p_x;"], list(in_signal.code()))

    def test_vectors_of_different_width_dont_connect(self):
        in_signal = LogicInVectorSignal(basename="x", width=3)
        out_signal = LogicOutVectorSignal(basename="x", width=2)
        in_signal.connect(out_signal)
        self.assertTrue(in_signal.is_missing_inputs())

    def test_signals_do_not_reconnect(self):
        in_signal = LogicInSignal(basename="x")
        out_signal = LogicOutSignal(basename="x").with_prefix("out")
        in_signal.connect(out_signal)
        out_signal = LogicOutSignal(basename="x").with_prefix("second_out")
        in_signal.connect(out_signal)
        self.assertEqual(["x <= out_x;"], list(in_signal.code()))


if __name__ == "__main__":
    unittest.main()
