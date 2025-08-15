-- This is the hard_sigmoid implementation for fixed point data
-- it has to use DSP slices to finish the arithmetic computation
-- Prefetching data is necessary since this layer is clocked
-- Version: 1.0
-- Created by: Chao
-- Last modified date: 2022.11.06

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity TestHardSigmoid_6_2 is
    generic (
        DATA_WIDTH : integer := 6;
        FRAC_WIDTH : integer := 2;
        ONE : integer := 4;
        ZERO_THRESHOLD : integer := -12;
        ONE_THRESHOLD : integer := 12;
        SLOPE : integer := 0;
        Y_INTERCEPT: integer := 2
    );
    port (
        enable : in std_logic;
    	clock  : in std_logic;
    	x      : in std_logic_vector(DATA_WIDTH-1 downto 0);
    	y      : out std_logic_vector(DATA_WIDTH-1 downto 0)
    );
end entity TestHardSigmoid_6_2;

architecture rtl of TestHardSigmoid_6_2 is
    signal fxp_input : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal fxp_output : signed(DATA_WIDTH-1 downto 0) := (others=>'0');

    constant fxp_slop : signed(DATA_WIDTH-1 downto 0) := to_signed(SLOPE, DATA_WIDTH);
    constant fxp_y_intercept : signed(DATA_WIDTH-1 downto 0) := to_signed(Y_INTERCEPT, DATA_WIDTH);

    -----------------------------------------------------------
    -- functions
    -----------------------------------------------------------
    function linear_op(a : in signed(DATA_WIDTH-1 downto 0);
                    x : in signed(DATA_WIDTH-1 downto 0);
                    b : in signed(DATA_WIDTH-1 downto 0)
            ) return signed is

        variable TEMP : signed(DATA_WIDTH*2-1 downto 0) := (others=>'0');
        variable TEMP2 : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
        variable TEMP3 : signed(FRAC_WIDTH-1 downto 0) := (others=>'0');
    begin
        TEMP := a * x;

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

    main_process : process (enable, clock)
    begin
        if (enable = '0') then
            fxp_output <= to_signed(0, DATA_WIDTH);
        elsif (rising_edge(clock)) then

            if fxp_input <= to_signed(ZERO_THRESHOLD, DATA_WIDTH) then
                fxp_output <= to_signed(0, DATA_WIDTH);
            elsif fxp_input >= to_signed(ONE_THRESHOLD, DATA_WIDTH) then
                fxp_output <= to_signed(ONE, DATA_WIDTH);
            else
                fxp_output <= linear_op(fxp_input, fxp_slop, fxp_y_intercept);
            end if;
        end if;
    end process;
end architecture rtl;
