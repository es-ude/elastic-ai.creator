----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 02/22/2025 01:52:23 PM
-- Design Name: 
-- Module Name: unified_activation_buffer_tb - rtl
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.bus_package.all;

entity unified_activation_buffer_tb is
end unified_activation_buffer_tb;

architecture rtl of unified_activation_buffer_tb is

    signal finished : std_logic := '0';
    constant half_period : time := 5ns;
    constant clk_period : time := 10ns;

    signal clk : std_logic := '0';
    signal nRST : std_logic := '1';
    signal enable : std_logic := '0';
    signal num_valid_vals : std_logic_vector(4 downto 0) := "00000";
    signal addr_out : std_logic_vector(4 downto 0); -- 5 Bit für 32 Adressen
    signal data_out : bus_array_4_8;
    signal r_w : std_logic; -- 1 für Read, 0 für Write
    signal addr_in : std_logic_vector(4 downto 0);
    signal data_in : bus_array_4_8;
    
    type t_test_data is array(0 to 31) of std_logic_vector(7 downto 0);

    constant test_data : t_test_data := (
        "01001001", "01011011", "10100101", "11110000",
        "00001111", "00110011", "11001100", "01101010",
        "10010110", "00011100", "11100011", "01010101",
        "10101010", "10000001", "01111110", "11000011",
        "00111100", "00010010", "10011001", "01101100",
        "11010011", "10110110", "01001010", "00101001",
        "11111111", "00000000", "01100110", "10011001",
        "11001001", "00101100", "01010110", "10101001"
    );

    -- 20 neue Werte für den zweiten Schreibvorgang
    type t_extra_data is array(0 to 19) of std_logic_vector(7 downto 0);

    constant extra_data : t_extra_data := (
        "11100000", "00011111", "11010101", "00110011",
        "10101010", "01010101", "10011001", "01101110",
        "00010000", "11101111", "00100100", "11000011",
        "01111000", "00000111", "10100110", "01001011",
        "01110011", "10000110", "11110001", "00001110"
    );

begin

    dut : entity work.unified_buffer(rtl)
    generic map(
        MAX_FEATURE_NUM => 32,
        X_ADDR_WIDTH => 5
    )
    port map(
        clk => clk,
        nRST => nRST,
        enable => enable,
        num_valid_vals => num_valid_vals,
        r_addr => addr_out,
        r_data => data_out,
        r_w => r_w,
        w_addr => addr_in,
        w_data => data_in
    );

    clk <= not clk after half_period when finished /= '1' else '0';

    stimulus : process
        variable test_addr : integer;
    begin
        -- Reset halten
        nRST <= '1';
        wait for clk_period;
        nRST <= '0';
        wait for 2*clk_period;
        nRST <= '1';
        wait for clk_period;

        enable <= '1';
        num_valid_vals <= std_logic_vector(to_unsigned(32-1, 5));
        r_w <= '0'; --schreiben

        -- **Schreiben der ersten 32 Werte**
        for i in 0 to 7 loop
            addr_in <= std_logic_vector(to_unsigned(i*4, 5));
            data_in(0 to 3) <= bus_array_4_8(test_data(i*4 to i*4+3));
            wait for clk_period;
        end loop;

        r_w <= '1'; -- Lesen

        -- **Lesen der ersten 32 Werte**
        for i in 0 to 7 loop
            addr_out <= std_logic_vector(to_unsigned(i*4, 5));
            wait for clk_period;
        end loop;

        r_w <= '0'; --Schreiben
        num_valid_vals <= std_logic_vector(to_unsigned(10-1,5));

        -- **Schreiben von 10 neuen Werten**
        for i in 0 to 2 loop
            addr_in <= std_logic_vector(to_unsigned(i*4, 5));
            wait for clk_period;
            data_in(0 to 3) <= bus_array_4_8(extra_data(i*4 to i*4+3));
            wait for clk_period;
        end loop;
        
        r_w <= '1'; -- Lesen
        
        -- **Lesen von 12 Werten um Zero Padding zu testen**
        for i in 0 to 2 loop
            addr_out <= std_logic_vector(to_unsigned(i*4, 5));
            wait for clk_period;
        end loop;

        -- Test beenden
        finished <= '1';
        enable <= '0';
        wait;
    end process;

end rtl;

