----------------------------------------------------------------------------------
-- Company: 
-- Engineer: Christopher Ringhofer
-- 
-- Create Date: 08.07.2018 17:55:48
-- Design Name: 
-- Module Name: uart_rxtx - Behavioral
-- Project Name: FPGA Weather Station
-- Target Devices: Digilent Arty S7-50 (Xilinx Spartan-7)
-- Tool Versions: 
-- Description: Implements the top-level UART module.
-- 
-- Dependencies: uart_tx, uart_rx, baudrate_gen
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity uart_rxtx is
    generic(
        clk_freq    : integer     := 100_000_000;   -- frequency of system clock in Hertz
        baud_rate   : integer     := 9_600;         -- data link baud rate in bits/second
        os_rate		: integer     := 16;            -- oversampling rate (in samples per baud period)
        d_width		: integer     := 8;             -- data bus width
        stop_bits   : integer     := 1;             -- number of stop bits
        use_parity	: integer     := 0;             -- 0 for no parity, 1 for parity
        parity_eo	: std_logic   := '0');          -- '0' for even, '1' for odd parity
    port(
        clk      : in std_logic;                                -- system clock
        reset_n  : in std_logic;                                -- asynchronous reset
        tx_en    : in std_logic;                                -- initiates transmission, latches in transmit data
        tx_data  : in std_logic_vector(d_width-1 downto 0);     -- data to transmit
        rx		 : in std_logic;							    -- receive pin
        tx       : out std_logic;                               -- transmit pin
        tx_busy  : out std_logic;                               -- transmission in progress
        rx_done  : out std_logic;                               -- data reception finished
        rx_error : out std_logic;                               -- start, parity, or stop bit error detected
        rx_data  : out std_logic_vector(d_width-1 downto 0));   -- data received
end uart_rxtx;

architecture Behavioral of uart_rxtx is

component baudrate_gen is
    generic(
        clk_freq    : integer := 100_000_000;   -- frequency of system clock in Hertz
        baud_rate   : integer := 9_600;         -- data link baud rate in bits/second
        os_rate     : integer := 16);           -- oversampling rate to find center of receive bits (in samples per baud period)
    port(
        clk         : in std_logic;     -- system clock
        reset_n     : in std_logic;     -- asynchronous reset
        baud_pulse  : out std_logic;    -- periodic pulse that occurs at the baud rate
        os_pulse    : out std_logic);   -- periodic pulse that occurs at the oversampling rate
end component;

component uart_tx is
    generic(
        d_width		: integer   := 8;    -- data bus width
        stop_bits   : integer   := 1;    -- number of stop bits
        use_parity  : integer   := 0;    -- 0 for no parity, 1 for parity
        parity_eo   : std_logic := '0'); -- '0' for even, '1' for odd parity
    port(
        clk     : in std_logic;                             -- system clock
        clk_en  : in std_logic;                             -- clock enable indicating baud pulses
        reset_n : in std_logic;                             -- asynchronous reset
        tx_en   : in std_logic;                             -- initiates transmission, latches in transmit data
        tx_data : in std_logic_vector(d_width-1 downto 0);  -- data to transmit
        tx_busy : out std_logic;                            -- transmission in progress
        tx      : out std_logic);                           -- transmit pin
end component;

component uart_rx is
    generic(
        d_width		: integer     := 8;    -- data bus width
        stop_bits   : integer     := 1;    -- number of stop bits
        use_parity  : integer     := 0;    -- 0 for no parity, 1 for parity
        parity_eo   : std_logic   := '0';  -- '0' for even, '1' for odd parity
        os_rate     : integer     := 16);  -- oversampling rate (in samples per baud period)
    port(
        clk         : in std_logic;                                 -- system clock
        clk_en      : in std_logic;                                 -- clock enable indicating oversampling pulses
        reset_n     : in std_logic;                                 -- asynchronous reset
        rx          : in std_logic;                                 -- receive pin
        rx_done     : out std_logic;                                -- data reception finished
        rx_error    : out std_logic;                                -- start, parity, or stop bit error detected
        rx_data     : out std_logic_vector(d_width-1 downto 0));    -- data received
end component;

signal baud_pulse   : std_logic := '0';
signal os_pulse     : std_logic := '0';

begin

baud_module : baudrate_gen
    generic map(
        clk_freq    => clk_freq,
        baud_rate   => baud_rate,
        os_rate     => os_rate)
    port map(
        clk         => clk,
        reset_n     => reset_n,
        baud_pulse  => baud_pulse,
        os_pulse    => os_pulse);
        
tx_module : uart_tx
    generic map(
        d_width     => d_width,
        stop_bits   => stop_bits,
        use_parity  => use_parity,
        parity_eo   => parity_eo)
    port map(
        clk     => clk,
        clk_en  => baud_pulse,
        reset_n => reset_n,
        tx_en   => tx_en,
        tx_data => tx_data,
        tx_busy => tx_busy,
        tx      => tx);
 
 rx_module : uart_rx
    generic map(
        d_width     => d_width,
        stop_bits   => stop_bits,
        use_parity  => use_parity,
        parity_eo   => parity_eo,
        os_rate     => os_rate)

    port map(
        clk         => clk,
        clk_en      => os_pulse,
        reset_n     => reset_n,
        rx          => rx,
        rx_done     => rx_done,
        rx_error    => rx_error,
        rx_data     => rx_data);

end Behavioral;
