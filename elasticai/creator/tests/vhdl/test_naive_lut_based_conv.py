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
 'generic (',
 'INPUT_WIDTH : integer := 1;',
 'OUTPUT_WIDTH : integer := 1',
 ');',
 'port (',
 'x : in std_logic_vector(INPUT_WIDTH-1 downto 0);',
 'y : out std_logic_vector(OUTPUT_WIDTH-1 downto 0)',
 ');',
 'end entity NaiveLUTConv;',
 'architecture id of NaiveLUTConv is',
 'begin',
 'y <="0" when x="1" else',
'"10" when x="10" ;',
 'end architecture id;'
        ]
        self.assertEqual(expected, code)


