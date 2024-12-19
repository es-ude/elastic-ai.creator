library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;
use work.counter_pkg.clog2;

entity padding_remover_tb is
end entity;


architecture behav of padding_remover_tb is
  constant NUM_INPUT_BYTES : natural := 4;
  constant NUM_OUTPUT_BITS : natural := 26;
  constant DATA_WIDTH: integer := 13;
  constant MAX_SIM_STEPS: integer := 20;
  constant BYTE : natural := 8;
  signal input_bytes :  std_logic_vector(NUM_INPUT_BYTES * 8 - 1 downto 0) := x"FAAA0BBB";
  signal d_in : std_logic_vector(NUM_INPUT_BYTES * BYTE - 1 downto 0);
  signal d_out : std_logic_vector(NUM_OUTPUT_BITS - 1 downto 0);
  signal one : std_logic := '1';
  signal zero : std_logic := '0';
  signal clk : std_logic := '1';
  signal sim_step : integer range 0 to MAX_SIM_STEPS;
  signal sim_step_v : std_logic_vector(clog2(MAX_SIM_STEPS+1) - 1 downto 0);
  signal padded_output : std_logic_vector(32 - 1 downto 0) := (others => 'X');
  signal single_bit_padded : std_logic_vector(8*4 - 1 downto 0) := (others => 'X');
  signal single_bit : std_logic_vector(4 - 1 downto 0) := b"1010";
begin

  sim_step <= to_integer(unsigned(sim_step_v));

  clk <= not clk after 10 ps;

  sim_step_counter_i : entity work.counter(rtl)
    generic map (MAX_VALUE => MAX_SIM_STEPS)
    port map (
      d_out => sim_step_v,
      enable => one,
      rst => zero,
      clk => clk
    );

  dut_i : entity work.padding_remover(rtl)
    generic map (
      DATA_WIDTH => DATA_WIDTH,
      DATA_DEPTH => 2
    )
    port map (
      d_in => input_bytes,
      d_out => d_out
    );

  padder_i : entity work.padder(rtl)
    generic map (
      DATA_WIDTH => 13,
      DATA_DEPTH => 2
      )
    port map (
      d_in => d_out,
      d_out => padded_output
);

    single_bit_padder_i : entity work.padder(rtl)
    generic map (
        DATA_WIDTH => 1,
        DATA_DEPTH => 4
    )
    port map (
      d_in => single_bit,
      d_out => single_bit_padded
    );


  process (clk) is
    constant expected_unpadded : std_logic_vector(26 - 1 downto 0) := (b"1" & x"AAA" & b"0" & x"BBB");
    constant expected_padded : std_logic_vector(32 - 1 downto 0) := x"1AAA0BBB";
    constant expected_single_bit_padded : std_logic_vector(32 - 1 downto 0) := x"01000100";
    begin
      if rising_edge(clk) then
        if sim_step = MAX_SIM_STEPS then
          finish;
        else
          assert single_bit_padded = expected_single_bit_padded
            report "expected single bit padded to " & to_hstring(expected_single_bit_padded) & " but was " & to_hstring(single_bit_padded) severity error;
          assert d_out = expected_unpadded
            report "expected " & to_hstring(expected_unpadded) & " but was " & to_hstring(d_out)  severity error;
          assert padded_output = expected_padded
            report "failed to pad output, expected " & to_hstring(expected_padded) & " but was " & to_hstring(padded_output) severity error;
        end if;
      end if;
    end process;



end architecture;
