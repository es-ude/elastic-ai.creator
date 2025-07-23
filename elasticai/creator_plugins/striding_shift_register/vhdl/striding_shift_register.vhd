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
    signal stride_count : natural range 0 to STRIDE - 1 := 0;
    signal last_intern_valid_in : std_logic;
    signal intern_valid_out : std_logic;
  begin


    valid_out <= intern_valid_out and last_intern_valid_in;

    process (stride_count, valid_in) is begin
      if stride_count = 0 and valid_in = '1' then
        intern_valid_in <= '1';
      else
        intern_valid_in <= '0';
      end if;
    end process;

    process(clk) is begin
     if rising_edge(clk) then
      last_intern_valid_in <= intern_valid_in;
     end if;
    end process;

    process (clk, rst) is begin
      if rst = '1' then
        stride_count <= 0;
      elsif rising_edge(clk) then
        if valid_in = '1' then
          if stride_count < STRIDE - 1 then
            stride_count <= stride_count + 1;
          else
            stride_count <= 0;
          end if;
        end if;
      end if;
    end process;

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
        valid_out => intern_valid_out
      );

end architecture;
