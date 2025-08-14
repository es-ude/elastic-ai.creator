from typing import cast

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.fixed_point.hard_sigmoid.layer import HardSigmoid


def test_vhdl_code_matches_expected_hardsigmoid() -> None:
    expected = """-- This is the hard_sigmoid implementation for fixed point data
-- it has to use DSP slices to finish the arithmetic computation
-- Prefetching data is necessary since this layer is clocked
-- Version: 1.1
-- Created by: Chao
-- Modified by: AE
-- Last modified date: 2022.11.06, 2025.08.14

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity sigmoid is
    generic (
        DATA_WIDTH : integer := 16;
        FRAC_WIDTH : integer := 8;
        ONE : integer := 256;
        ZERO_THRESHOLD : integer := -768;
        ONE_THRESHOLD : integer := 768;
        SLOPE : integer := 42;
        Y_INTERCEPT: integer := 128
    );
    port (
        enable : in std_logic;
    	clock  : in std_logic;
    	x      : in std_logic_vector(DATA_WIDTH-1 downto 0);
    	y      : out std_logic_vector(DATA_WIDTH-1 downto 0)
    );
end entity sigmoid;

architecture rtl of sigmoid is
    signal fxp_input : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal fxp_output : signed(DATA_WIDTH-1 downto 0) := (others=>'0');

    constant fxp_slop : signed(DATA_WIDTH-1 downto 0) := to_signed(SLOPE, DATA_WIDTH);
    constant fxp_y_intercept : signed(DATA_WIDTH-1 downto 0) := to_signed(Y_INTERCEPT, DATA_WIDTH);

    -----------------------------------------------------------
    -- functions
    -----------------------------------------------------------
    function linear_op(a : in signed(DATA_WIDTH-1 downto 0);
                    x0 : in signed(DATA_WIDTH-1 downto 0);
                    b : in signed(DATA_WIDTH-1 downto 0)
            ) return signed is

        variable TEMP : signed(DATA_WIDTH*2-1 downto 0) := (others=>'0');
        variable TEMP2 : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
        variable TEMP3 : signed(FRAC_WIDTH-1 downto 0) := (others=>'0');
    begin
        TEMP := a * x0;

        TEMP2 := TEMP(DATA_WIDTH+FRAC_WIDTH-1 downto FRAC_WIDTH);
        TEMP3 := TEMP(FRAC_WIDTH-1 downto 0);
        if TEMP2(DATA_WIDTH-1) = '1' and TEMP3 /= 0 then
            TEMP2 := TEMP2 + 1;
        end if;

        if TEMP>0 and TEMP2<0 then
            TEMP2 := ('0', others => '1');
        elsif TEMP<0 and TEMP2>0 then
            TEMP2 := ('1', others => '0');
        end if;
        return TEMP2 + b;
    end function;

begin

    fxp_input <= signed(x);
    y <= std_logic_vector(fxp_output);

    main_process : process (clock)
    begin
        if (enable = '0') then
            fxp_output <= to_signed(0, DATA_WIDTH);
        else
            if (rising_edge(clock)) then
                if fxp_input <= to_signed(ZERO_THRESHOLD, DATA_WIDTH) then
                    fxp_output <= to_signed(0, DATA_WIDTH);
                elsif fxp_input >= to_signed(ONE_THRESHOLD, DATA_WIDTH) then
                    fxp_output <= to_signed(ONE, DATA_WIDTH);
                else
                    fxp_output <= linear_op(fxp_input, fxp_slop, fxp_y_intercept);
                end if;
            end if;
        end if;
    end process;
end architecture rtl;
""".splitlines()
    sigmoid = HardSigmoid(total_bits=16, frac_bits=8)
    build_path = InMemoryPath("build", parent=None)
    design = sigmoid.create_design("sigmoid")
    design.save_to(build_path)
    actual = cast(InMemoryFile, build_path["sigmoid"]).text
    for text in actual:
        print(text)
    assert actual == expected
