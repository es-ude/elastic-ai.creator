----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 02/24/2025 01:32:36 PM
-- Design Name: 
-- Module Name: DPU_TB - rtl
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

entity DPU_TB is
end DPU_TB;

architecture rtl of DPU_TB is

    signal clk : std_logic := '0';
    constant half_period : time := 5ns;
    constant clk_period : time := 10ns;
    signal finished : std_logic := '0';
    
    signal nRST : std_logic;
    signal enable : std_logic;
    signal x_addr : std_logic_vector(5-1 downto 0);
    signal x_bus  : bus_array_4_8;
    signal w_addr : std_logic_vector(10-1 downto 0);
    signal w_bus  : bus_array_4_8;
    signal o_addr : std_logic_vector(5-1 downto 0);
    signal o_bus : bus_array_4_8;
    
    signal done : std_logic;
    signal en : std_logic;

begin

    clk <= not clk after half_period when finished /= '1' else '0';

    dut : entity work.DPU(rtl)
    generic map(
        X_ADDR_WIDTH => 5,
        IN_FEATURE_NUM => 32,
        W_ADDR_WIDTH => 10,
        NUM_LAYER => 3,
        NETWORK_DIMENSIONS => (32, 20, 14, 3),
        WEIGHT_NUM => 999
    )
    port map(
        clk => clk,
        nRST => nRST,
        enable => enable,
        x_addr => x_addr,
        x_bus => x_bus,
        w_addr => w_addr,
        w_bus => w_bus,
        o_addr => o_addr,
        o_bus => o_bus,
        done => done
    );
    
    wram : entity work.weight_bram(rtl)
    port map(
        clk => clk,
        en => en,
        addr => w_addr,
        data => w_bus
    );
    
    aram : entity work.activations_bram(rtl)
    port map(
        clk => clk,
        en => en,
        addr => x_addr,
        data => x_bus
    );
    
    stimulus : process
    begin
    
        nRST <= '1';
        wait for clk_period;
        nRST <= '0';
        wait for 2*clk_period;
        nRST <= '1';
        wait for clk_period;
        
        en <= '1';
        
        wait for clk_period;
        enable <= '1';
        wait until done = '1';
        enable <= '0';
        wait for clk_period;
        
        finished <= '1';
    
    end process;

end rtl;
