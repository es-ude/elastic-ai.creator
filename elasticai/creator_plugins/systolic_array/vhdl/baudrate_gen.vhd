----------------------------------------------------------------------------------
-- Company: 
-- Engineer: Christopher Ringhofer
-- 
-- Create Date: 08.07.2018 17:50:28
-- Design Name: 
-- Module Name: baudrate_gen - Behavioral
-- Project Name: FPGA Weather Station
-- Target Devices: Digilent Arty S7-50 (Xilinx Spartan-7)
-- Tool Versions: 
-- Description: Implements a baud rate generator for driving UART transmit and receive modules.
--              The baud_pulse is meant to drive the clock_enable pin of a UART transmitter,
--              the os_pulse is meant to drive the clock_enable pin of a UART receiver for oversampling.
-- 
-- Dependencies: None
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity baudrate_gen is
    generic(
        clk_freq    : integer := 100_000_000;   -- frequency of system clock in Hertz
        baud_rate	: integer := 9_600;		    -- data link baud rate in bits/second
        os_rate		: integer := 16);           -- oversampling rate to find center of receive bits (in samples per baud period)
    port(
        clk		    : in std_logic;     -- system clock
        reset_n     : in std_logic;     -- asynchronous reset
        baud_pulse  : out std_logic;    -- periodic pulse that occurs at the baud rate
        os_pulse    : out std_logic);   -- periodic pulse that occurs at the oversampling rate
end baudrate_gen;

architecture Behavioral of baudrate_gen is

begin

    -- generate clock enable pulses at the baud rate and the oversampling rate
    BAUD_GEN : process(reset_n, clk)
    
        variable count_baud : integer range 0 to clk_freq/baud_rate-1           := 0; -- counter to determine baud rate period
        variable count_os   : integer range 0 to clk_freq/baud_rate/os_rate-1   := 0; -- counter to determine oversampling period
    
    begin
    
        if reset_n = '1' then      -- asynchronous reset asserted
            baud_pulse  <= '0';
            os_pulse    <= '0';
            count_baud  := 0;
            count_os    := 0;
            
        elsif rising_edge(clk) then
        
            -- create oversampling enable pulse
            if count_os < clk_freq/baud_rate/os_rate-1 then -- oversampling period not reached yet
                count_os := count_os + 1;
                os_pulse <= '0';    
            else                                            -- oversampling period reached
                os_pulse <= '1';
                count_os := 0;
            end if;
        
            -- create baud enable pulse
            if count_baud < clk_freq/baud_rate-1 then   -- baud period not reached yet
                baud_pulse <= '0';
                count_baud := count_baud + 1;
            else                                        -- baud period reached
                baud_pulse  <= '1';
                count_baud  := 0;
                count_os    := 0;                       -- reset oversample counter to prevent accumulated error
            end if;
        
        end if;
        
    end process BAUD_GEN;

end Behavioral;
