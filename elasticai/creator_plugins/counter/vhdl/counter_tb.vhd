----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 10/06/2024 01:04:09 PM
-- Design Name: 
-- Module Name: counter_tb - Behavioral
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
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.counter_pkg.all;
use std.env.all;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity counter_tb is
--  Port ( );
end counter_tb;

architecture Behavioral of counter_tb is
    signal rst : std_logic := '0';
    signal enable : std_logic := '0';
    signal d_counter : std_logic_vector(4 - 1 downto 0);
    signal clk : std_logic := '0';
    signal wrap_arounds : unsigned(8 -1 downto 0) := to_unsigned(0, 8);
begin

    clk <= not clk after 10 ns;


    dut_i : entity work.counter(rtl) 
        generic map (MAX_VALUE => 9)
        port map ( clk => clk, enable => enable, d_out => d_counter, rst => rst);
        

   
   
   process is
   begin
        rst <= '1';
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        rst <= '0';
        wait until rising_edge(clk);
        enable <= '1';
       wait until rising_edge(enable);
       for i in 0 to 9 loop
           wait until rising_edge(clk);
           assert d_counter = std_logic_vector(to_unsigned(i, d_counter'length));
       end loop;
       wait until rising_edge(clk);
       wait until rising_edge(clk);
       rst <= '1';
       wait until rising_edge(clk);
       assert d_counter = std_logic_vector(to_unsigned(0, d_counter'length)) report "expected counter to be zero but was " & to_string(d_counter) severity error;
       wait until rising_edge(clk);
       assert d_counter = std_logic_vector(to_unsigned(0, d_counter'length)) report "expected counter to be zero but was " & to_string(d_counter) severity error;
       rst <= '0';
       wait until rising_edge(clk);
      assert d_counter = std_logic_vector(to_unsigned(0, d_counter'length)) report "expected counter to be zero but was " & to_string(d_counter) severity error;

       wait;
   end process;
   
   process (clk) is
   begin
     if falling_edge(clk) then
       if unsigned(d_counter) = 0 then
         wrap_arounds <= wrap_arounds + 1;
       end if;   
     end if;
   end process;
   
   process is
   begin
     for i in 0 to 20 loop
       wait until rising_edge(clk);
     end loop;
     assert wrap_arounds > to_unsigned(1, wrap_arounds'length) report "expected more than one wraparound, but found " & to_string(wrap_arounds);
     finish;
   end process;
        


end Behavioral;
