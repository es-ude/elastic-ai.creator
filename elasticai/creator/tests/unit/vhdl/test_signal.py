import unittest

from elasticai.creator.vhdl.code import Code
from elasticai.creator.vhdl.signals.base_signal import BaseInSignal, Identifiable


class MyOutSignal(Identifiable):
    def id(self) -> str:
        return "my_out_signal"


class AlwaysMatchingSignal(BaseInSignal):
    def id(self) -> str:
        return "alwaysmatch"

    def definition(self) -> str:
        return ""

    def matches(self, other: Identifiable) -> bool:
        return True


class NeverMatchingSignal(BaseInSignal):
    def id(self) -> str:
        return "nevermatch"

    def definition(self) -> str:
        return ""

    def matches(self, other: Identifiable) -> bool:
        return False


class SignalTestCase(unittest.TestCase):
    def test_never_matching_signal_is_missing_inputs_after_connect(self):
        in_signal = NeverMatchingSignal()
        in_signal.connect(MyOutSignal())
        self.assertTrue(in_signal.is_missing_inputs())

    def test_always_matching_signal_is_not_missing_inputs_after_connect(self):
        in_signal = AlwaysMatchingSignal()
        in_signal.connect(MyOutSignal())
        self.assertFalse(in_signal.is_missing_inputs())

    def test_code_produces_correct_vhdl_line_after_connecting(self):
        in_signal = AlwaysMatchingSignal()
        out_signal = MyOutSignal()
        in_signal.connect(out_signal)
        self.assertEqual([f"{in_signal.id()} <= {out_signal.id()};"], in_signal.code())


if __name__ == "__main__":
    unittest.main()
