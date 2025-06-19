library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;
use ieee.math_real.all;
use work.padding_pkg.all;
library vunit_lib;
context vunit_lib.vunit_context;

entity padding_and_buffers_tb is
  generic (runner_cfg: string);
end entity;


architecture behav of padding_and_buffers_tb is
  constant DATA_DEPTH : integer := 3;
  constant DATA_WIDTH : integer := 2;

  

  signal buffer_ready_in : std_logic;
  signal buffer_ready_out : std_logic;
  signal buffer_valid_in : std_logic;
  signal buffer_valid_out : std_logic;

  subtype buffer_d_t is std_logic_vector(DATA_WIDTH - 1 downto 0);
  subtype byte_t is std_logic_vector(7 downto 0);
  signal buffer_d_in : buffer_d_t := (others => 'X');
  signal buffer_d_out : buffer_d_t := (others => 'X');
  
  signal bytes_d_in : byte_t;
  signal bytes_d_out : byte_t;

  signal bytes_valid_in : std_logic;
  signal bytes_valid_out : std_logic;
  signal bytes_ready_in : std_logic := '0';
  signal bytes_ready_out : std_logic;

  signal clk : std_logic := '0';
  signal rst : std_logic := '0';

  constant half_period : time := 1 fs;
  constant period : time := 2*half_period;

  
begin

  unpadder : entity work.dynamic_padding_remover(rtl)
    generic map (
      DATA_WIDTH => DATA_WIDTH
    )
    port map (
      d_in => bytes_d_in,
      d_out => buffer_d_in,
      valid_in => bytes_valid_in,
      ready_in => buffer_ready_out,
      valid_out => buffer_valid_in,
      ready_out => bytes_ready_out,
      clk => clk,
      rst => rst
    );

  fifobuffer : entity work.fifo_buffer(rtl)
    generic map (
      DATA_WIDTH => DATA_WIDTH,
      DATA_DEPTH => DATA_DEPTH
    )
    port map (
      d_in => buffer_d_in,
      d_out => buffer_d_out,
      valid_in => buffer_valid_in,
      valid_out => buffer_valid_out,
      ready_in => buffer_ready_in,
      ready_out => buffer_ready_out,
      clk => clk,
      rst => rst
    );

  padder : entity work.dynamic_padder(rtl)
    generic map (
      DATA_WIDTH => DATA_WIDTH
    )
    port map (
      d_in => buffer_d_out,
      d_out => bytes_d_out,
      valid_in => buffer_valid_out,
      valid_out => bytes_valid_out,
      ready_in => bytes_ready_in,
      ready_out => buffer_ready_in,
      clk => clk,
      rst => rst
    );

    clk <= not clk after half_period;

  process is  
    type data_t is array(0 to DATA_DEPTH - 1) of byte_t;
    constant input_d: data_t := (x"FF", x"00", x"F1");
    constant expected_d : data_t := (x"03", x"00", x"01");
  begin
    test_runner_setup(runner, runner_cfg);
    bytes_valid_in <= '1';
    for i in 0 to DATA_DEPTH - 1 loop
      if bytes_ready_out = '1' then
        bytes_d_in <= input_d(i);
        wait for period;
      end if;
    end loop;
    bytes_valid_in <= '0';
    bytes_ready_in <= '1';
    wait for period;
    for i in 0 to DATA_DEPTH - 1 loop
      wait for period;
      check_equal(bytes_valid_out, '1');
      check_equal(bytes_d_out, expected_d(i));
    end loop;
    wait for period;
    bytes_ready_in <= '0';
    wait for period;
    check_equal(bytes_valid_out, '0');

      

    test_runner_cleanup(runner);
  end process;
end architecture;
