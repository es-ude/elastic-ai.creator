import numpy as np

from elasticai.creator.tests.vhdl.vhdl_file_testcase import GeneratedVHDLCodeTest
from elasticai.creator.vhdl.generator.rom import Rom
from elasticai.creator.vhdl.number_representations import (
    FloatToSignedFixedPointConverter,
)


class GenerateROMVhdTest(GeneratedVHDLCodeTest):
    def test_compare_files(self) -> None:
        rom_name = "rom_bi"
        data_width = 12
        frac_width = 4
        # biases for the input gate
        Bi = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
        floats_to_signed_fixed_point_converter = FloatToSignedFixedPointConverter(
            bits_used_for_fraction=frac_width, strict=False
        )
        values = [floats_to_signed_fixed_point_converter(x) for x in Bi]

        generate_rom = Rom(
            rom_name=rom_name,
            data_width=data_width,
            values=values,
            resource_option="auto",
        )
        generated_code = list(generate_rom())

        expected_code = [
            "library ieee;",
            "use ieee.std_logic_1164.all;",
            "use ieee.std_logic_unsigned.all;",
            "entity rom_bi is",
            "port (",
            "clk : in std_logic;",
            "en : in std_logic;",
            "addr : in std_logic_vector(3-1 downto 0);",
            "data : out std_logic_vector(12-1 downto 0)",
            ");",
            "end entity rom_bi;",
            "architecture rtl of rom_bi is",
            "type rom_bi_array_t is array (0 to 2**3-1) of std_logic_vector(12-1 downto 0);",
            'signal ROM : rom_bi_array_t:=(x"011",x"023",x"034",x"046",x"058",x"069",x"000",x"000");',
            "attribute rom_style : string;",
            'attribute rom_style of ROM : signal is "auto";',
            "begin",
            "ROM_process: process(clk)",
            "begin",
            "if rising_edge(clk) then",
            "if (en = '1') then",
            "data <= ROM(conv_integer(addr));",
            "end if;",
            "end if;",
            "end process ROM_process;",
            "end architecture rtl;",
        ]

        self.assertEqual(expected_code, generated_code)
