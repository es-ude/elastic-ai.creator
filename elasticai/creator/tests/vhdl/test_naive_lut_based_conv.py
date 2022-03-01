import unittest

from unittest import TestCase

from elasticai.creator.vhdl.generator.lut_conv import NaiveLUTBasedConv
from elasticai.creator.vhdl.number_representations import BitVector


class TestNaiveLUTBasedConv(TestCase):
    def test_NaiveLUTBasedConv(self):
        generate_conv = NaiveLUTBasedConv(
            component_name="id",
            input_width= 1,
            output_width=1,
            inputs=[[BitVector(1,1,1)],[BitVector(2,2,2)]],
            outputs=[[BitVector(0,0,0)],[BitVector(2,2,2)]],
        )
        code = list(generate_conv())
        expected = [
 'library ieee;',
 'use ieee.std_logic_1164.all;',
 'use ieee.numeric_std.all;',
 'entity NaiveLUTConv is',
 '\tgeneric (',
 '\t\tINPUT_WIDTH : integer := 1;',
 '\t\tOUTPUT_WIDTH : integer := 1',
 '\t);',
 '\tport (',
 '\t\tx : in std_logic_vector(INPUT_WIDTH-1 downto 0);',
 '\t\ty : out std_logic_vector(OUTPUT_WIDTH-1 downto 0)',
 '\t);',
 'end entity NaiveLUTConv;',
 'architecture id of NaiveLUTConv is',
 'begin',
 '\ty <="0" when x="1" else',
'\t"10" when x="10" ;',
 'end architecture id;'
        ]
        self.assertEqual(expected, code)


