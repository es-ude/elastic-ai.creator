library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;
library vunit_lib;
context vunit_lib.vunit_context;

entity padding_remover_tb is
  generic (runner_cfg: string);
end entity;


architecture behav of padding_remover_tb is
  constant BYTE : natural := 8;
  constant NUM_INPUT_BYTES : natural := 4;
  constant PADDING_REMOVER_WIDTH: integer := 2*BYTE - 3;
  constant PADDING_REMOVER_DEPTH: integer := 2;
  signal d_in:  std_logic_vector(NUM_INPUT_BYTES * BYTE - 1 downto 0) := (others => 'X');

  signal padding_remover_d_out : std_logic_vector(
      PADDING_REMOVER_WIDTH * PADDING_REMOVER_DEPTH - 1 downto 0
      ) := (others => 'X');

  signal padder_d_out : std_logic_vector(
      PADDING_REMOVER_DEPTH * 2 * BYTE  - 1 downto 0
      ) := (others => 'X');

  constant SINGLE_BIT_PADDER_DEPTH : integer := 4;

  signal single_bit_padder_d_out : std_logic_vector(
      SINGLE_BIT_PADDER_DEPTH * BYTE - 1 downto 0
      ) := (others => 'X');

  signal clk : std_logic := '1';
  constant clk_freq : time := 2 ps;
  constant clk_cycle : time := 2 * clk_freq;
begin


  clk <= not clk after clk_freq;


  dut_i : entity work.padding_remover(rtl)
    generic map (
      DATA_WIDTH => PADDING_REMOVER_WIDTH,
      DATA_DEPTH => PADDING_REMOVER_DEPTH
    )
    port map (
      d_in => d_in,
      d_out => padding_remover_d_out
    );

  padder_i : entity work.padder(rtl)
    generic map (
      DATA_WIDTH => PADDING_REMOVER_WIDTH,
      DATA_DEPTH => PADDING_REMOVER_DEPTH
      )
    port map (
      d_in => d_in(padding_remover_d_out'range),
      d_out => padder_d_out
  );

    single_bit_padder_i : entity work.padder(rtl)
    generic map (
        DATA_WIDTH => 1,
        DATA_DEPTH => 4
    )
    port map (
      d_in => d_in(3 downto 0),
      d_out => single_bit_padder_d_out
    );


    process is

      procedure compare_single_bit_padding(
          constant input: std_logic_vector(3 downto 0);
          constant expected: std_logic_vector(single_bit_padder_d_out'range)
      ) is
      begin
        d_in(3 downto 0) <= input;
        wait for clk_cycle;
        check_equal(single_bit_padder_d_out, expected);
      end procedure;

      procedure padder_reverses_remover(constant input: std_logic_vector(d_in'range)) is
      begin
        d_in <= input;
        wait for clk_cycle;
        d_in(padding_remover_d_out'range) <= padding_remover_d_out;
        wait for clk_cycle;
        check_equal(padder_d_out, input);
      end procedure;
      
      procedure check_padding_remover(
          constant input: std_logic_vector(d_in'range);
          constant expected: std_logic_vector(padding_remover_d_out'range)
      ) is
      begin
        d_in <= input;
        wait for clk_cycle;
        check_equal(padding_remover_d_out, expected);
      end procedure;

      procedure check_padder(
          constant input: std_logic_vector(padding_remover_d_out'range);
          constant expected: std_logic_vector(d_in'range)
      ) is
      begin
        d_in(padding_remover_d_out'range) <= input;
        wait for clk_cycle;
        check_equal(padder_d_out, expected);
      end procedure;
  begin
    test_runner_setup(runner, runner_cfg);
    if run("compare single padded bit") then
      compare_single_bit_padding(b"0101", x"00_01_00_01");
      compare_single_bit_padding(b"1101", x"01_01_00_01");
      compare_single_bit_padding(b"1010", x"01_00_01_00");
    end if;

    if run("padder reverses remover") then
      padder_reverses_remover(x"1F_00_1F_00");
      padder_reverses_remover(x"1A_BC_13_35");
    end if;

    if run("check padder") then
      check_padder(b"1" & x"F_FF" & b"1" & x"A_BB", x"1F_FF_1A_BB");
    end if;

    if run("check padding remover") then
      check_padding_remover(x"FF_FF_FF_FF", b"1" & x"F_FF" & b"1" & x"F_FF");
    end if;

    test_runner_cleanup(runner);
  end process;



end architecture;
