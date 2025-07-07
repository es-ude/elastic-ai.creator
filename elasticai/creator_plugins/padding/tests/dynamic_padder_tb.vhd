library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;
use ieee.math_real.all;
use work.padding_pkg.all;
library vunit_lib;
context vunit_lib.vunit_context;

entity dynamic_padding_tb is
  generic (runner_cfg: string);
end entity;


architecture behav of dynamic_padding_tb is
    
  constant BYTE : natural := 8;
  constant NUM_INPUT_BYTES : natural := 4;
  constant DATA_WIDTH: integer := 2*BYTE - 3;
  constant DATA_DEPTH: integer := 2;
  signal remover_d_in:  std_logic_vector(BYTE - 1 downto 0) := (others => 'X');

  signal remover_d_out : std_logic_vector(
      DATA_WIDTH - 1 downto 0
      ) := (others => 'X');

  signal unpadded_v : std_logic_vector(DATA_DEPTH*DATA_WIDTH - 1 downto 0);

  signal padded_v : std_logic_vector(
    NUM_INPUT_BYTES*8 - 1 downto 0
  );

  signal padder_valid_in : std_logic := '0';
  signal padder_valid_out : std_logic := '0';
  signal padder_d_in : std_logic_vector(remover_d_out'range);
  signal padder_d_out : std_logic_vector(remover_d_in'range);

  signal clk : std_logic := '1';
  constant clk_freq : time := 2 ps;
  constant clk_cycle : time := 2 * clk_freq;
  signal rst : std_logic := '0';
  signal ready_in : std_logic := '1';
  signal ready_out : std_logic := '0';
  signal padder_ready : std_logic := '0';
  signal remover_valid_in : std_logic := '0';
  signal remover_valid_out : std_logic := '0';
begin

  clk <= not clk after clk_freq;


  dut_i : entity work.dynamic_padding_remover(rtl)
    generic map (
      DATA_WIDTH => DATA_WIDTH
    )
    port map (
      ready_in => ready_in,
      ready_out => ready_out,
      valid_in => remover_valid_in,
      valid_out => remover_valid_out,
      rst => rst,
      clk => clk,
      d_in => remover_d_in,
      d_out => remover_d_out
    );

  padder : entity work.dynamic_padder(rtl)
    generic map (
      DATA_WIDTH => DATA_WIDTH
    )
    port map (
      ready_in => ready_in,
      ready_out => padder_ready,
      valid_in => padder_valid_in,
      valid_out => padder_valid_out,
      rst => rst,
      clk => clk,
      d_in => padder_d_in,
      d_out => padder_d_out
      
    );

    write_unpadded_p:
    process (clk) is
      variable data_id : integer range 0 to DATA_DEPTH := 0;
    begin
      if rising_edge(clk) then
        read_from_padding_component(
          data_id => data_id,
          rst => rst,
          data_buffer => unpadded_v,
          d_out => remover_d_out,
          valid_out => remover_valid_out,
          DATA_DEPTH => DATA_DEPTH,
          DATA_WIDTH => DATA_WIDTH
        );
      end if;

    end process;

    read_padded_p:
    process (clk) is
      variable byte_id : integer range 0 to NUM_INPUT_BYTES := 0;
    begin
      if rising_edge(clk) then
        read_from_padding_component(
          data_id => byte_id,
          rst => rst,
          data_buffer => padded_v,
          d_out => padder_d_out,
          DATA_DEPTH => NUM_INPUT_BYTES,
          DATA_WIDTH => 8,
          valid_out => padder_valid_out
        );
      end if;
    end process;
    
    process is
      
      procedure check_padding_remover(
          constant input: std_logic_vector(NUM_INPUT_BYTES*8 - 1 downto 0);
          constant expected: std_logic_vector(unpadded_v'range)
      ) is
      begin
        remover_valid_in <= '1';
        for i in NUM_INPUT_BYTES-1 downto 0 loop
          remover_d_in <= input(BYTE*(i+1) - 1 downto 8*i);
          wait for clk_cycle;
        end loop;
        remover_valid_in <= '0';
        wait for clk_cycle;
        rst <= '1';
        wait for clk_cycle;
        rst <= '0';
        wait for clk_cycle;
        check_equal(unpadded_v, expected);
     end procedure;

    procedure check_padder(
      constant input: in std_logic_vector(DATA_DEPTH*DATA_WIDTH - 1 downto 0);
      constant expected: in std_logic_vector(padded_v'range)
    ) is
    begin
      padder_valid_in <= '1';
      for j  in 0 to DATA_DEPTH - 1 loop
        padder_d_in <= input(DATA_WIDTH*(j+1) - 1 downto DATA_WIDTH*j);
      wait until padder_ready = '1';
      end loop;
      wait for clk_cycle;
      padder_valid_in <= '0';
      wait for clk_cycle;
      rst <= '1';
      wait for clk_cycle;
      rst <= '0';
      wait for clk_cycle;
      check_equal(padded_v, expected);

    end procedure;

  begin
    test_runner_setup(runner, runner_cfg);
    rst <= '0';
    remover_valid_in <= '0';
    padder_valid_in <= '0';

    if run("check padding remover") then
      check_padding_remover(x"FF_FF_FF_FF", b"1" & x"F_FF" & b"1" & x"F_FF");
    end if;

    wait for clk_cycle;

    if run("check padder") then
      check_padder(b"1" & x"F_FF" & b"1" & x"F_FF", x"1F_FF_1F_FF");
    end if;

    test_runner_cleanup(runner);
  end process;



end architecture;
