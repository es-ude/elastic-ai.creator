from typing import cast

from elasticai.creator.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.hard_tanh.layer import FPHardTanh


def test_vhdl_code_matches_expected() -> None:
    expected = """-- This is the hard_sigmoid implementation for fixed point data
-- followed by the logic from pytorch:
-- https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html
-- Version: 1.0
-- Created by: Chao
-- Last modified date: 2023.01.31

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tanh is
    generic (
        DATA_WIDTH : integer := 16;
        FRAC_WIDTH : integer := 8;
        MIN_VAL : integer := -256;
        MAX_VAL : integer := 256
    );
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector(DATA_WIDTH-1 downto 0);
        y      : out std_logic_vector(DATA_WIDTH-1 downto 0)
    );
end entity tanh;

architecture rtl of tanh is
    signal fp_input : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal fp_output : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
begin
    fp_input <= signed(x);
    y <= std_logic_vector(fp_output);

    main_process : process (enable, clock)
    begin
        if (enable = '0') then
            fp_output <= to_signed(0, DATA_WIDTH);
        elsif (rising_edge(clock)) then

            if fp_input <= to_signed(MIN_VAL, DATA_WIDTH) then
                fp_output <= to_signed(MIN_VAL, DATA_WIDTH);
            elsif fp_input >= to_signed(MAX_VAL, DATA_WIDTH) then
                fp_output <= to_signed(MAX_VAL, DATA_WIDTH);
            else
                fp_output <= fp_input;
            end if;
        end if;
    end process;
end architecture rtl;
""".splitlines()
    tanh = FPHardTanh(total_bits=16, frac_bits=8)
    build_path = InMemoryPath("build", parent=None)
    design = tanh.translate("tanh")
    design.save_to(build_path)
    actual = cast(InMemoryFile, build_path["tanh"]).text
    assert actual == expected
