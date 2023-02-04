-- This is the hard_sigmoid implementation for fixed point data
-- followed by the logic from pytorch:
-- https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html
-- Version: 1.0
-- Created by: Chao
-- Last modified date: 2023.01.31

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity fp_hard_tanh_${layer_name} is
generic (
    DATA_WIDTH : integer := ${data_width};
    FRAC_WIDTH : integer := ${frac_width};
    MIN_VAL : integer := ${min_val};
    MAX_VAL : integer := ${max_val}
);

port (
    enable : in std_logic;
    clock  : in std_logic;
    input  : in std_logic_vector(DATA_WIDTH-1 downto 0);
    output : out std_logic_vector(DATA_WIDTH-1 downto 0)
);
end entity fp_hard_tanh_${layer_name};

architecture rtl of fp_hard_tanh_${layer_name} is
    signal fp_input : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal fp_output : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
begin

    fp_input <= signed(input);
    output <= std_logic_vector(fp_output);

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
