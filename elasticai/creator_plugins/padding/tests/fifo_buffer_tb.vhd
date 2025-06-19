library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;
use ieee.math_real.all;
library vunit_lib;
context vunit_lib.vunit_context;

entity fifo_buffer_tb is
  generic (runner_cfg: string);
end entity;


architecture behav of fifo_buffer_tb is
  constant DATA_WIDTH : positive := 4;
  constant DATA_DEPTH : positive := 5;
  
  signal d_in : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => 'X');
  signal d_out : std_logic_vector(DATA_WIDTH - 1 downto 0);
  signal valid_in : std_logic := '0';
  signal valid_out : std_logic;
  signal ready_in : std_logic := '0';
  signal ready_out : std_logic;
  signal clk : std_logic := '1';
  signal rst : std_logic := '0';

  constant half_period : time := 1 fs;
  constant period : time := 2*half_period;
begin

  dut_0 : entity work.fifo_buffer(rtl)
  generic map (
    DATA_WIDTH => DATA_WIDTH,
    DATA_DEPTH => DATA_DEPTH
  )
  port map (
    d_in => d_in,
    d_out => d_out,
    clk => clk,
    rst => rst,
    valid_in => valid_in,
    ready_in => ready_in,
    ready_out => ready_out,
    valid_out => valid_out
  );

  clk <= not clk after half_period;


  process is
    variable expected : std_logic_vector(d_out'range);
    
    procedure write_ABCD is
    begin
      d_in <= x"A";
      valid_in <= '1';
      wait for period;
      d_in <= x"B";
      wait for period;
      d_in <= x"C";
      wait for period;
      d_in <= x"D";
      valid_in <= '0';
      wait for period;
    end procedure;

    procedure reset is
    begin
      rst <= '1';
      wait for period;
      rst <= '0';
      wait for period;
    end procedure;

  begin
    test_runner_setup(runner, runner_cfg);

    reset;

    if run("buffer starts ready") then
      check_equal(ready_out, '1');
    end if;

    if run("buffer starts invalid") then
      check_equal(valid_out, '0');
    end if;


    if run("buffer is valid after writing once") then
      valid_in <= '1';
      wait for period;
      wait for period;
      valid_in <= '0';
      check_equal(valid_out, '0');
      wait for period;
      check_equal(valid_out, '1');
      wait for period;
    end if;

    if run("buffer is not valid after writing four and reading four") then
      write_ABCD;
      ready_in <= '1';
      check_equal(valid_out, '1');
      wait for period;
      check_equal(valid_out, '1');
      wait for period;
      check_equal(valid_out, '1');
      wait for period;
      check_equal(valid_out, '1');
      wait for period;
      check_equal(valid_out, '0');
      ready_in <= '0';
      wait for period;
    end if;

    if run("can write and read ABCD") then
      write_ABCD;
      ready_in <= '1';
      wait for period;
      expected := x"A";
      check_equal(d_out, expected);
      wait for period;
      expected := x"B";
      check_equal(d_out, expected);
      wait for period;
      expected := x"C";
      check_equal(d_out, expected);
      wait for period;
      expected := x"D";
      check_equal(d_out, expected);
      ready_in <= '0';
      wait for period;
    end if;

    if run("buffer full after writing 5 times in a row") then
      valid_in <= '1';
      wait for period;
      wait for period;
      wait for period;
      wait for period;
      wait for period;
      valid_in <= '0';
      check_equal(ready_out, '0');
      wait for period;
    end if;
    
    test_runner_cleanup(runner);
  end process;
  

end architecture;
