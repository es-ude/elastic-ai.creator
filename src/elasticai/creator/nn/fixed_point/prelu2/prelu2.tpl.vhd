-- //////////////////////////////////////////////////////////////////////////////////
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

entity ${layer_name} is
    generic (
        DATA_WIDTH  : integer := ${data_width};
        SCALING     : integer := ${scaling}
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
