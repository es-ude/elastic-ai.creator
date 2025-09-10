----------------------------------------------------------------------------------
-- Company: 
-- Engineer: Christopher Ringhofer
-- 
-- Create Date: 08.07.2018 17:51:38
-- Design Name: 
-- Module Name: uart_rx - Behavioral
-- Project Name: FPGA Weather Station
-- Target Devices: Digilent Arty S7-50 (Xilinx Spartan-7)
-- Tool Versions: 
-- Description: Implements the receiver side of the UART protocol.
-- 
-- Dependencies: None
-- 
-- Revision:
-- Revision 0.02 - Added majority vote
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity uart_rx is
    generic(
        d_width		: integer     := 8;    -- data bus width
        stop_bits   : integer     := 1;    -- number of stop bits
        use_parity  : integer     := 0;    -- 0 for no parity, 1 for parity
        parity_eo   : std_logic   := '0';  -- '0' for even, '1' for odd parity
        os_rate		: integer     := 16);  -- oversampling rate (in samples per baud period)
    port(
        clk         : in std_logic;                                 -- system clock
        clk_en      : in std_logic;                                 -- clock enable indicating oversampling pulses
        reset_n     : in std_logic;                                 -- asynchronous reset
        rx		    : in std_logic;									-- receive pin
        rx_done     : out std_logic;								-- data reception finished
        rx_error    : out std_logic;								-- start, parity, or stop bit error detected
        rx_data	    : out std_logic_vector(d_width-1 downto 0));    -- data received
end uart_rx;

architecture Behavioral of uart_rx is
 
    type rx_machine is (idle, start, data, parity, stop);                                       -- receive state machine data type
    signal rx_state     : rx_machine := idle;                                                   -- receive state machine
    signal parity_error : std_logic;                                                            -- receive parity error flag
    signal rx_parity    : std_logic_vector(d_width downto 0);                                   -- calculation of receive parity
    signal rx_buffer    : std_logic_vector(use_parity+d_width-1 downto 0) := (others => '0');   -- values received
    
    signal rx_synced        : std_logic_vector(1 downto 0); -- shift register for rx data synchronization
    signal rx_majority_vote : std_logic;                    -- the majority bit value of three consecutive samples

begin
    
    -- double-register the incoming data for synchronization
    -- this removes problems caused by metastability and allows it to be used in the UART_RX clock domain
    SAMPLE : process (clk)
    begin
        if rising_edge(clk) then
            rx_synced <= rx_synced(0) & rx;
        end if;
    end process SAMPLE;
    
    
    -- aggregate three consecutive samples and perform a majority vote
    VOTE : process (clk, clk_en)
    
        variable rx_aggregated : std_logic_vector(2 downto 0) := "111";
        
    begin
    
        if rising_edge(clk) AND clk_en = '1' then
            rx_aggregated := rx_aggregated(1 downto 0) & rx_synced(1);
        end if;
        
        rx_majority_vote <= (rx_aggregated(0) and (rx_aggregated(1) or rx_aggregated(2)))
                            or (rx_aggregated(1) and rx_aggregated(2));
        
    end process VOTE;


    -- receive state machine
	UART_RX : process (reset_n, clk, clk_en)
	
        variable rx_count :	integer range 0 to d_width-1 := 0;   -- count the bits received
        variable os_count :	integer range 0 to os_rate-1 := 0;   -- count the oversampling rate pulses
        
	begin
		
		if reset_n = '1' then             -- asynchronous reset asserted
            os_count := 0;                -- clear oversampling pulse counter
			rx_count := 0;                -- clear receive bit counter
			rx_done  <= '0';              -- clear receive done signal
			rx_error <= '0';              -- clear receive errors
			rx_data  <= (others => '0');  -- clear received data output
			rx_state <= idle;             -- put in idle state
			
		elsif rising_edge(clk) and clk_en = '1' then      -- enable clock at oversampling rate
		
			case rx_state is
			
			when idle =>
                rx_done     <= '0';         -- clear receive done flag
                rx_error    <= '0';
                if rx_synced(1) = '0' then  -- start bit might be present
                    rx_state <= start;
                else
                    rx_state <= idle;
                end if;
                
            when start =>
                if os_count < os_rate/2 - 1 then    -- oversampling pulse counter is not at start bit center
                    if rx_synced(1) = '0' then
                        os_count := os_count + 1;
                        rx_state <= start;
                    else							-- start bit not present
                        os_count := 0;
                        rx_state <= idle;           -- back to idle state
                    end if;
                else							    -- oversampling pulse counter is at bit center
                    os_count    := 0;
                    rx_state    <= data;		    -- advance to data state
                end if;
                
            when data =>
                if os_count < os_rate-1 then                    -- not center of bit
                    os_count := os_count + 1;                   -- increment oversampling pulse counter
                    rx_state <= data;
                else                                            -- center of bit
                    rx_buffer(rx_count) <= rx_majority_vote;    -- get majority voted rx value and store bit in receive buffer
                    os_count := 0;
                    if rx_count < d_width-1 then                -- not all bits received yet
                        rx_count := rx_count + 1;
                        rx_state <= data;
                    else
                        rx_count  := 0;
                        if use_parity = 1 then 
                            rx_state <= parity;
                        else
                            rx_state <= stop;
                        end if;
                    end if;
                end if;
                
            when parity =>
                if os_count < os_rate-1 then    -- not center of bit
                    os_count := os_count + 1;
                    rx_state <= parity;
                else
                    rx_buffer(use_parity+d_width-1) <= rx_majority_vote;    -- add parity bit to receive buffer
                    os_count := 0;
                    rx_state <= stop;
                end if;
                
            when stop =>
                if os_count < os_rate-1 then    -- not center of bit
                    os_count := os_count + 1;
                    rx_state <= stop;
                else                            -- center of bit
                    os_count := 0;
                    if rx_count < stop_bits-1 then    -- not all stop bits received yet
                        rx_count := rx_count + 1;
                        rx_state <= stop;
                    else
                        rx_data     <= rx_buffer(d_width-1 downto 0);           -- output data received to user logic
                        rx_error    <= parity_error or not rx_majority_vote;    -- output parity / stop bit error flag
                        rx_done     <= '1';                                     -- deassert received busy flag
                        rx_state    <= idle;                                    -- return to idle state
                    end if;
                end if;

            end case;
        
		end if;
		
	end process UART_RX;
		
	--receive parity calculation logic
	rx_parity(0) <= parity_eo;
	rx_parity_logic : for i in 0 to d_width-2 generate
		rx_parity(i+1) <= rx_parity(i) xor rx_buffer(i+1);
	end generate;
	
	with use_parity select  -- compare calculated parity bit with received parity bit to determine error
		parity_error <= rx_parity(d_width) xor rx_buffer(use_parity+d_width-1) when 1,
		                                                                 '0' when others;

end Behavioral;
