-- This is the hard_sigmoid implementation for fixed point data
-- followed by the logic from pytorch:
-- https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html
-- Version: 1.0
-- Created by: Chao
-- Last modified date: 2023.01.31

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ${layer_name} is
    generic (
        DATA_WIDTH : integer := ${data_width};
        FRAC_WIDTH : integer := ${frac_width};
        MIN_VAL : integer := ${min_val};
        MAX_VAL : integer := ${max_val}
    );
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector(DATA_WIDTH-1 downto 0);
        y      : out std_logic_vector(DATA_WIDTH-1 downto 0)
    );
end entity ${layer_name};

architecture rtl of ${layer_name} is
    signal fxp_input : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal fxp_output : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
begin
    fxp_input <= signed(x);
    y <= std_logic_vector(fxp_output);

    main_process : process (enable, clock)
    begin
        if (enable = '0') then
            fxp_output <= to_signed(0, DATA_WIDTH);
        elsif (rising_edge(clock)) then

            if fxp_input <= to_signed(MIN_VAL, DATA_WIDTH) then
                fxp_output <= to_signed(MIN_VAL, DATA_WIDTH);
            elsif fxp_input >= to_signed(MAX_VAL, DATA_WIDTH) then
                fxp_output <= to_signed(MAX_VAL, DATA_WIDTH);
            else
                fxp_output <= fxp_input;
            end if;
        end if;
    end process;
end architecture rtl;
