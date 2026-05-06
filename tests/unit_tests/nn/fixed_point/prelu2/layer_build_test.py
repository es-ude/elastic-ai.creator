from typing import cast

import elasticai.creator.nn.fixed_point as nn_creator
from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath


def test_vhdl_code_matches_expected_prelu2() -> None:
    expected = """-- //////////////////////////////////////////////////////////////////////////////////
-- Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
-- Engineer:        AE
--
-- Create Date:     22.01.2026, 08:20:44
-- Last modified:   05.05.2026, 20:37
-- Module Name:     Programmable ReLU-Activation Function for DNN
-- Target Devices:  ASIC / FPGA
-- Tool Versions:   1v0
-- Processing:      LUT-based processing
-- Dependencies:    None
--
-- State: 	        Works!
-- Improvements:    None
-- Parameters:      DATA_WIDTH --> Bitwidth of input data
--                  SCALING --> Number of bits for bit-shifting negative values
-- ////////////////////////////////////////////////////////////////////////////////

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity prelu2 is
    generic (
        DATA_WIDTH  : integer := 8;
        SCALING     : integer := -64;
    );
    port (
        enable : in std_logic;
    	x      : in std_logic_vector(DATA_WIDTH-1 downto 0);
    	y      : out std_logic_vector(DATA_WIDTH-1 downto 0)
    );
end entity prelu2;

architecture rtl of prelu2 is
    signal fxp_input : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal fxp_output : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
begin
    fxp_input <= signed(x);
    y <= std_logic_vector(fxp_output);

    process (fxp_input) begin
        if (enable = '0') then
            fxp_output <= to_signed(0, DATA_WIDTH);
        else
            if fxp_input < 0 then
                fxp_output <= shift_right(fxp_input, SCALING);
            else
                fxp_output <= fxp_input;
            end if;
        end if;
    end process;
end architecture rtl;
""".splitlines()
    actfunc = nn_creator.PReLU2(
        total_bits=8,
        frac_bits=5,
        init=0.25,
    )
    build_path = InMemoryPath("build", parent=None)
    design = actfunc.create_design("prelu2")
    design.save_to(build_path)
    actual = cast(InMemoryFile, build_path["prelu2"]).text
    for text in actual:
        print(text)
    assert actual == expected
