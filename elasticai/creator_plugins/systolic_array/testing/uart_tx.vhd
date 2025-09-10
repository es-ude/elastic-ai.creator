----------------------------------------------------------------------------------
-- Company: 
-- Engineer: Christopher Ringhofer
-- 
-- Create Date: 08.07.2018 17:53:37
-- Design Name: 
-- Module Name: uart_tx - Behavioral
-- Project Name: FPGA Weather Station
-- Target Devices: Digilent Arty S7-50 (Xilinx Spartan-7)
-- Tool Versions: 
-- Description: Implements the transmitter side of the UART protocol.
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

entity uart_tx is
    generic(
        d_width		: integer     := 8;    -- data bus width
        stop_bits   : integer     := 1;    -- number of stop bits
        use_parity	: integer     := 0;    -- 0 for no parity, 1 for parity
        parity_eo	: std_logic   := '0'); -- '0' for even, '1' for odd parity
    port(
        clk     : in std_logic;                             -- system clock
        clk_en  : in std_logic;                             -- clock enable indicating baud pulses
        reset_n : in std_logic;                             -- asynchronous reset
        tx_en   : in std_logic;                             -- initiates transmission, latches in transmit data
        tx_data : in std_logic_vector(d_width-1 downto 0);  -- data to transmit
        tx_busy : out std_logic;                            -- transmission in progress
        tx      : out std_logic);							-- transmit pin
end uart_tx;

architecture Behavioral of uart_tx is
  
    type    tx_machine is (idle, data, parity, stop, cleanup);                       -- transmit state machine data type
    signal	tx_state     : tx_machine := idle;                                       -- transmit state machine
    signal	tx_parity    : std_logic_vector(d_width downto 0);                       -- calculation of transmit parity
    signal  tx_buffer    : std_logic_vector(d_width downto 0) := (others => '0');    -- values to be transmitted + start bit
    
    signal  busy_internal   : std_logic := '0';    -- internal busy state

begin

    UART_TX : process(reset_n, clk)
    
        variable tx_count : integer range 0 to d_width := 0; -- transmitted bits count
        
    begin
    
        if reset_n = '1' then   -- asynchronous reset asserted
            tx_count        := 0;
            tx              <= '1';
            busy_internal   <= '1'; -- set transmit busy signal to indicate unavailability
            tx_state        <= idle;
        
        elsif rising_edge(clk) then
        
            if (tx_en = '1' AND busy_internal = '0') then   -- load TX data
                tx_count        := 0;
                busy_internal   <= '1';
                tx_buffer       <= tx_data & '0';   -- load data into transmit buffer
                tx_state        <= data;            -- change to data transfer state
                
            elsif clk_en = '1' then    -- only when clock enable pulse is asserted
                            
                case tx_state is
                
                when idle =>
                    busy_internal   <= '0';
                    tx              <= '1';         -- idle high
                    tx_count        := 0;
                    tx_state        <= idle;
                    
                when data =>
                    tx <= tx_buffer(tx_count);      -- put bit onto tx line
                    if tx_count < d_width then      -- not all bits transmitted yet
                        tx_count := tx_count + 1;
                        tx_state <= data;
                    else                            -- all bits transmitted
                        tx_count := 0;
                        if use_parity = 1 then
                            tx_state <= parity;     -- add parity bit if specified
                        else
                            tx_state <= stop;       
                        end if;
                    end if;
                    
                when parity =>
                    tx_count    := 0;
                    tx          <= tx_parity(d_width);  -- add parity from last position of register
                    tx_state    <= stop;
                    
                when stop =>
                    tx <= '1';
                    if tx_count < stop_bits-1 then  -- not all stop bits transmitted yet
                        tx_count := tx_count + 1;
                        tx_state <= stop;
                    else
                        tx_count        := 0;
                        busy_internal   <= '0';
                        tx_state        <= idle;
                    end if;
                    
                when others =>
                    tx_state <= idle;
                    
                end case;
            end if;
        end if;
        
    end process UART_TX;
    
    tx_busy <= busy_internal;
    
    -- transmit parity calculation logic, highest position indicates parity
    tx_parity(0) <= parity_eo;
    tx_parity_logic : for i in 0 to d_width-1 generate  -- multiple instances of same circuitry
        tx_parity(i+1) <= tx_parity(i) XOR tx_data(i);
    end generate;
    
end Behavioral;
