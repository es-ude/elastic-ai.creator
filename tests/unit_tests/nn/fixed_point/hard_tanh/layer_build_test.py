from typing import cast

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.fixed_point.hard_tanh.layer import HardTanh


def test_vhdl_code_matches_expected_hardtanh() -> None:
    expected = """-- This is the hard_tanh implementation for fixed point data
-- followed by the logic from pytorch:
-- https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html
-- Version: 1.1
-- Created by: Chao
-- Changed by: AE
-- Last modified date: 2025.08.11

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
    signal fxp_input : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal fxp_output : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
begin
    fxp_input <= signed(x);
    y <= std_logic_vector(fxp_output);

    main_process : process(fxp_input)
    begin
        if (enable = '0') then
            fxp_output <= to_signed(0, DATA_WIDTH);
        else
            if fxp_input < to_signed(MIN_VAL, DATA_WIDTH) then
                fxp_output <= to_signed(MIN_VAL, DATA_WIDTH);
            elsif fxp_input > to_signed(MAX_VAL, DATA_WIDTH) then
                fxp_output <= to_signed(MAX_VAL, DATA_WIDTH);
            else
                fxp_output <= fxp_input;
            end if;
        end if;
    end process;
end architecture rtl;
""".splitlines()
    tanh = HardTanh(total_bits=16, frac_bits=8)
    build_path = InMemoryPath("build", parent=None)
    design = tanh.create_design("tanh")
    design.save_to(build_path)
    actual = cast(InMemoryFile, build_path["tanh"]).text
    assert actual == expected
