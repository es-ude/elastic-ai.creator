library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.counter_pkg.all;


entity sliding_window is
    generic (
        INPUT_WIDTH : positive;
        OUTPUT_WIDTH : positive;
        STRIDE : positive := 1
);
    port (
        signal clk : in std_logic;
        signal d_in : in std_logic_vector(INPUT_WIDTH - 1 downto 0);
        signal d_out : out std_logic_vector(OUTPUT_WIDTH - 1 downto 0);
        signal valid_in : in std_logic;
        signal valid_out : out std_logic;
        signal rst : in std_logic
);
end entity;

architecture rtl of sliding_window is
  constant MAX_VALUE : integer := (INPUT_WIDTH - OUTPUT_WIDTH)/STRIDE;
  signal counter : std_logic_vector(clog2(MAX_VALUE+1) - 1 downto 0) := (others => '0');
  signal counter_d : integer := 0;
  signal intern_valid : std_logic := '1';
begin
  counter_d <= to_integer(unsigned(counter));
  d_out <= d_in(INPUT_WIDTH - STRIDE * counter_d - 1 downto INPUT_WIDTH - STRIDE * counter_d - OUTPUT_WIDTH);
  valid_out <= valid_in and intern_valid;
  
  process (clk) is
      begin
          if rising_edge(clk) then
            if rst = '1' then
              intern_valid <= '1';
            end if;
            if valid_in = '1' then
              if counter_d = MAX_VALUE  then
                intern_valid <= '0';
            else
                intern_valid <= '1';
            end if;
          end if;
      end if;
  end process;
  
  counter_i : entity work.counter(rtl)
    generic map (
        MAX_VALUE => MAX_VALUE 
     )
     port map (
        clk => clk,
        rst => rst,
        enable => valid_in,
        d_out => counter
      );
end architecture;
