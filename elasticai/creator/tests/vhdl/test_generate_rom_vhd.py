import math
import numpy as np

from elasticai.creator.tests.vhdl.vhdl_file_testcase import GeneratedVHDLCodeTest
from elasticai.creator.vhdl.generator.generator_functions import float_array_to_string
from elasticai.creator.vhdl.generator.rom import Rom


class GenerateROMVhdTest(GeneratedVHDLCodeTest):
    def test_compare_files(self) -> None:
        rom_name = "rom_bi"
        data_width = 12
        frac_bits = 4
        # biases for the input gate
        Bi = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
        addr_width = math.ceil(math.log2(len(Bi)))
        array_value = float_array_to_string(float_array=Bi, frac_bits=frac_bits, nbits=data_width)

        generate_rom = Rom(
            rom_name=rom_name,
            data_width=data_width,
            addr_width=addr_width,
            array_value=array_value
        )
        generated_code = list(generate_rom())

        expected_code = [
            'library ieee;',
            'use ieee.std_logic_1164.all;',
            'use ieee.std_logic_unsigned.all;',
            'entity rom_bi is',
            'port (',
            'clk : in std_logic;',
            'en : in std_logic;',
            'addr : in std_logic_vector(3-1 downto 0);',
            'data : out std_logic_vector(12-1 downto 0)',
            ');',
            'end entity rom_bi;',
            'architecture rom_bi_rtl of rom_bi is',
            'type rom_bi_array_t is array (0 to 2**3-1) of std_logic_vector(12-1 downto 0);',
            'signal ROM : rom_bi_array_t:=(x"011",x"023",x"034",x"046",x"058",x"069",x"000",x"000");',
            'attribute rom_style : string;',
            'attribute rom_style of ROM : signal is "block";',
            'begin',
            'ROM_process: process(clk)',
            'begin',
            "if rising_edge(clk) then\nif (en = '1') then\ndata <= ROM(conv_integer(addr));",
            'end if;',
            'end if;',
            'end process ROM_process;',
            'end architecture rom_bi_rtl;'
        ]

        self.assertEqual(expected_code, generated_code)
