import unittest

from elasticai.creator.vhdl.generator.precomputed_scalar_function import (
    NaiveLUTBasedConv,
)
from unittest import TestCase


@unittest.SkipTest
class TestNaiveLUTBasedConv(TestCase):
    def test_basic_instantiation(self):
        generate_conv = NaiveLUTBasedConv(
            implements="implemented_entity",
            name="my_architecture",
            inputs=[],
            outputs=[],
        )
        code = list(generate_conv())
        expected = [
            "architecture my_architecture of implemented_entity is",
            "begin",
            "end my_architecture;",
        ]
        self.assertEqual(expected, code)

    def test_one_bit_output_version(self):
        generate_conv = NaiveLUTBasedConv(
            implements="", name="", inputs=["110"], outputs=["0"]
        )
        expected = ['\toutput <= "0" when input = "110";']
        code = list(generate_conv())
        code = code[2:]
        self.assertEqual(expected, code)
