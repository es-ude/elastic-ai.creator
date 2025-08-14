from typing import cast

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.fixed_point.relu.layer import ReLU


def test_vhdl_code_matches_expected() -> None:
    expected = """-- This is the ReLU implementation for fixed-point data
-- it only checks the highest bit of the input data
-- when the CLOCK_OPTION is enabled, please notice the data only updates until the clock arises.
-- Version: 1.1
-- Created by: Chao
-- Changed by: AE
-- Last modified date: 2025.08.11

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity relu is
    generic (
        DATA_WIDTH   : integer := 16;
        CLOCK_OPTION : boolean := true
    );
    port (
        enable : in std_logic;
    	clock  : in std_logic;
    	x      : in std_logic_vector(DATA_WIDTH-1 downto 0);
    	y      : out std_logic_vector(DATA_WIDTH-1 downto 0)
    );
end entity relu;

architecture rtl of relu is
    signal fxp_input : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal fxp_output : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
begin
    fxp_input <= signed(x);
    y <= std_logic_vector(fxp_output);

    clocked: if CLOCK_OPTION generate
        main_process : process (clock)
        begin
            if (enable = '0') then
                fxp_output <= to_signed(0, DATA_WIDTH);
            else
                if (rising_edge(clock)) then
                    if fxp_input < 0 then
                        fxp_output <= to_signed(0, DATA_WIDTH);
                    else
                        fxp_output <= fxp_input;
                    end if;
                end if;
            end if;
        end process;
    end generate;

    async: if (not CLOCK_OPTION) generate
        process (fxp_input)
        begin
            if (enable = '0') then
                fxp_output <= to_signed(0, DATA_WIDTH);
            else
                if fxp_input < 0 then
                    fxp_output <= to_signed(0, DATA_WIDTH);
                else
                    fxp_output <= fxp_input;
                end if;
            end if;
        end process;
    end generate;
end architecture rtl;
""".splitlines()
    relu = ReLU(total_bits=16, use_clock=True)
    build_path = InMemoryPath("build", parent=None)
    design = relu.create_design("relu")
    design.save_to(build_path)
    actual = cast(InMemoryFile, build_path["relu"]).text
    for text in actual:
        print(text)
    assert actual == expected
