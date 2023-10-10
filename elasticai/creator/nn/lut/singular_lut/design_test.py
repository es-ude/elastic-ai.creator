from creator.file_generation.virtual_path import VirtualPath
from lut.singular_lut.design import LUT


def test_lut_3_in_1_out():
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
            when others => "0";
        end case;
    end process;
end rtl;
"""
    lut = LUT(name="my_lut", in_bits=3, out_bits=1, outputs=[0, 0, 0, 0, 1, 1, 1, 1])
    build_dir = VirtualPath("root")
    lut.save_to(build_dir)
    actual = build_dir.children["my_lut.vhd"].read()
    assert expected == actual


def test_lut_2_in_2_out():
    expected = """library ieee;
use ieee.std_logic_1164.all;

entity my_lut is
    port (
        enable : in std_logic;
        clock : in std_logic;
        x : in std_logic_vector(2-1 downto 0);
        y : out std_logic_vector(2-1 downto 0);
    );
end;

architecture rtl of my_lut is
begin
    process (x)
    begin
        case x is
            when "00" => y <= "11";
            when "01" => y <= "10";
            when "10" => y <= "10";
            when "11" => y <= "01";
            when others => "00";
        end case;
    end process;
end rtl;
"""
    lut = LUT(
        name="my_lut", in_bits=2, out_bits=2, outputs=[(1, 1), (1, 0), (1, 0), (0, 1)]
    )
    build_dir = VirtualPath("root")
    lut.save_to(build_dir)
    actual = build_dir.children["my_lut.vhd"].read()
    assert expected == actual
