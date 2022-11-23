-- This is the ReLU implementation for fixed-point data
-- it only checks the highest bit of the input data
-- when the CLOCK_OPTION is enabled, please notice the data only updates until the clock arises.
-- Version: 1.0
-- Created by: Chao
-- Last modified date: 2022.11.06

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity fp_relu_${layer_name} is
generic (
    DATA_WIDTH : integer := ${data_width};
    CLOCK_OPTION : boolean := ${clock_option}
);
port (
    enable : in std_logic;
	clock  : in std_logic;
	input  : in std_logic_vector(DATA_WIDTH-1 downto 0);
	output : out std_logic_vector(DATA_WIDTH-1 downto 0)
);
end entity fp_relu_${layer_name};

architecture rtl of fp_relu_${layer_name} is

    signal fp_input : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal fp_output : signed(DATA_WIDTH-1 downto 0) := (others=>'0');

begin

    fp_input <= signed(input);
    output <= std_logic_vector(fp_output);

    clocked: if CLOCK_OPTION generate
        main_process : process (enable, clock)
        begin
            if (enable = '0') then
                fp_output <= to_signed(0, DATA_WIDTH);
            elsif (rising_edge(clock)) then

                if fp_input < 0 then
                    fp_output <= to_signed(0, DATA_WIDTH);
                else
                    fp_output <= fp_input;
                end if;
            end if;
        end process;
    end generate;

    async: if (not CLOCK_OPTION) generate
        process (enable, fp_input)
        begin
            if enable = '0' then
                fp_output <= to_signed(0, DATA_WIDTH);
            else
                if fp_input < 0 then
                    fp_output <= to_signed(0, DATA_WIDTH);
                else
                    fp_output <= fp_input;
                end if;
            end if;
        end process;
    end generate;

end architecture rtl;
