import unittest

from elasticai.creator.vhdl.language.signals import Signal, SignalBuilder


class LogicSignalTestCase(unittest.TestCase):
    """
    TODO:
    - make sure signals always have a default value
        - "'0'" for logic and "(other => '0')"
    """

    def test_logic_in_signal_definition_is_p_x(self) -> None:
        in_signal: Signal = SignalBuilder().id("x").build()
        self.assertEqual("signal p_x : std_logic;", in_signal.definition(prefix="p_"))

    def test_logic_in_signal_definition_is_y(self):
        in_signal = SignalBuilder().id("y").build()
        self.assertEqual("signal y : std_logic;", in_signal.definition())

    def test_logic_vector_in_signal_definition_is_x_0_downto_0(self):
        definition = SignalBuilder().id("x").width(1).build().definition()
        self.assertEqual("signal x : std_logic_vector(0 downto 0);", definition)

    def test_logic_vector_in_signal_definition_is_y_3_downto_0_with_default_and_prefix(
        self,
    ):
        definition = (
            SignalBuilder()
            .id("x")
            .width(4)
            .default("(other => '0')")
            .build()
            .definition()
        )
        self.assertEqual(
            "signal x : std_logic_vector(3 downto 0) := (other => '0');", definition
        )

    def test_connecting_signal_that_only_accepts_y_does_not_accept_x(self):
        in_signal = SignalBuilder().id("x").build()
        out_signal = SignalBuilder().id("x").accepted_names(["y"]).build()
        self.assertFalse(in_signal.accepts(out_signal))

    def test_vectors_of_different_width_dont_match(self):
        in_signal = SignalBuilder().id("x").width(2).build()
        out_signal = SignalBuilder().accepted_names(["x"]).width(3).build()
        self.assertFalse(in_signal.accepts(out_signal))

    def test_vector_does_not_match_non_vector_signal(self):
        vector = SignalBuilder().id("x").width(2).build()
        non_vector = SignalBuilder().accepted_names(["x"]).width(0).build()
        self.assertFalse(non_vector.accepts(vector))


if __name__ == "__main__":
    unittest.main()
