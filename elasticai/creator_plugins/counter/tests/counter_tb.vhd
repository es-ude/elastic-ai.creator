library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.counter_pkg.all;
use std.env.all;

library vunit_lib;
context vunit_lib.vunit_context;

entity counter_tb is
  generic (runner_cfg : string);
end counter_tb;

architecture Behavioral of counter_tb is
    signal rst : std_logic := '0';
    signal enable : std_logic := '0';
    signal d_counter : std_logic_vector(4 - 1 downto 0);
    signal clk : std_logic := '0';
    signal wrap_arounds : unsigned(8 -1 downto 0) := to_unsigned(0, 8);
    constant clk_freq : time := 2 ps;
    constant clk_cycle : time := 2 * clk_freq;
begin

    clk <= not clk after clk_freq;


    dut_i : entity work.counter(rtl) 
        generic map (MAX_VALUE => 9)
        port map ( clk => clk, enable => enable, d_out => d_counter, rst => rst);
        

   
   
  process is
    procedure count_to(constant number: natural) is
    begin
      rst <= '1';
      wait for clk_cycle;
      rst <= '0';
      enable <= '1';
      wait for number*clk_cycle;
      enable <= '0';
    end procedure;

    procedure check_counter(constant number: natural) is
      constant expected : std_logic_vector(d_counter'range) := std_logic_vector(to_unsigned(number, d_counter'length));
    begin
      check_equal(d_counter, expected);
    end procedure;

    procedure can_count_to(constant number: natural) is
    begin
      count_to(number);
      check_counter(number);
    end procedure;
      
   begin
      test_runner_setup(runner, runner_cfg);
      if run("can count to one") then
        can_count_to(1);
      end if;

      if run("can count to two") then
        can_count_to(2);
      end if;

      if run("can count to max value 9") then
        can_count_to(9);
      end if;

      if run("counter wraps at 9") then
        count_to(10);
        check_counter(0);
      end if;

      test_runner_cleanup(runner);
   end process;

end Behavioral;
