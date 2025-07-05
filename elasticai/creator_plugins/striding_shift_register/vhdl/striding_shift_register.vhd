library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.counter_pkg.all;

entity striding_shift_register is
      generic (
      DATA_WIDTH: positive;
      NUM_POINTS: positive;
      STRIDE: positive := 1  -- only write d on every STRIDEth rising edge,
                             -- ie., if
                             -- number_of_observed_rising_edges % STRIDE = 0
                             -- we use this to implement a layer with stride >
                             -- 1
    );
    port (
      rst : in std_logic;
      clk : in std_logic;
      d_in : in std_logic_vector(DATA_WIDTH - 1 downto 0);
      d_out : out std_logic_vector(DATA_WIDTH * NUM_POINTS - 1 downto 0);
      valid_in : in std_logic;
      valid_out : out std_logic
    );
end entity;

architecture rtl of striding_shift_register is
    signal intern_valid_in : std_logic := '1';
    signal counter : std_logic_vector(clog2(STRIDE) - 1 downto 0);
  begin


    intern_valid_in <= '1' when  (counter = std_logic_vector(to_unsigned(0, counter'length))) and valid_in = '1'
                    else '0';


    reg_i : entity work.shift_register
      generic map (
        DATA_WIDTH => DATA_WIDTH,
        NUM_POINTS => NUM_POINTS
      )
      port map (
        clk => clk,
        d_in => d_in,
        rst => rst,
        d_out => d_out,
        valid_in => intern_valid_in,
        valid_out => valid_out
      );

    counter_i : entity work.counter
      generic map (
        MAX_VALUE => (STRIDE - 1)
      )
      port map (
        clk => clk,
        d_out => counter,
        enable => valid_in,
        rst => rst
      );
end architecture;
