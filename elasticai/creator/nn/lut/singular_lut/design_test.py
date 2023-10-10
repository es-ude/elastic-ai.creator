from creator.file_generation.virtual_path import VirtualPath
from lut.singular_lut.design import LUT


def test_lut():
    expected = """library ieee;
use ieee.std_logic_1164.all;

entity my_lut is
    port (
        enable : in std_logic;
        clock : in std_logic;
        x : in std_logic_vector(3-1 downto 0);
        y : out std_logic_vector(1-1 downto 0);
    );
end;

architecture rtl of my_lut is
begin
    process (x)
    begin
        case x is
            when "000" => y <= "0";
            when "001" => y <= "0";
            when "010" => y <= "0";
            when "011" => y <= "0";
            when "100" => y <= "1";
            when "101" => y <= "1";
            when "110" => y <= "1";
            when "111" => y <= "1";
        end case;
    end process;
end rtl;
"""
    lut = LUT(name="my_lut", in_bits=3, out_bits=1, outputs=[0, 0, 0, 0, 1, 1, 1, 1])
    build_dir = VirtualPath("root")
    lut.save_to(build_dir)
    actual = build_dir.children["my_lut.vhd"].read()
    assert expected == actual
