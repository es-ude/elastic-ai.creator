import unittest

from elasticai.creator.vhdl.language.signals import Signal


class LogicSignalTestCase(unittest.TestCase):
    """
    TODO:
    - make sure signals always have a default value
        - "'0'" for logic and "(other => '0')"
    """

    def test_logic_in_signal_definition_is_p_x(self) -> None:
        in_signal: Signal = Signal(id="x", accepted_names=[], width=0)
        self.assertEqual(
            "signal p_x : std_logic := '0';", in_signal.definition(prefix="p_")
        )

    def test_logic_in_signal_definition_is_y(self):
        in_signal = Signal(id="y", accepted_names=[], width=0)
        self.assertEqual("signal y : std_logic := '0';", in_signal.definition())

    def test_logic_vector_in_signal_definition_is_x_0_downto_0(self):
        definition = Signal(id="x", width=1, accepted_names=[]).definition()
        self.assertEqual(
            "signal x : std_logic_vector(0 downto 0) := (other => '0');", definition
        )

    def test_logic_vector_in_signal_definition_is_y_3_downto_0_with_default_and_prefix(
        self,
    ):
        definition = Signal(id="x", width=4, accepted_names=[]).definition()
        self.assertEqual(
            "signal x : std_logic_vector(3 downto 0) := (other => '0');", definition
        )

    def test_connecting_signal_that_only_accepts_y_does_not_accept_x(self):
        in_signal = Signal(id="x", accepted_names=[], width=0)
        out_signal = Signal(id="x", accepted_names=["y"], width=0)
        self.assertFalse(in_signal.accepts(out_signal))

    def test_vectors_of_different_width_dont_match(self):
        in_signal = Signal(id="x", accepted_names=[], width=2)
        out_signal = Signal(id="x", accepted_names=["x"], width=3)
        self.assertFalse(in_signal.accepts(out_signal))

    def test_vector_does_not_match_non_vector_signal(self):
        vector = Signal(id="x", accepted_names=[], width=2)
        non_vector = Signal(id="x", accepted_names=["x"], width=0)
        self.assertFalse(non_vector.accepts(vector))


if __name__ == "__main__":
    unittest.main()
