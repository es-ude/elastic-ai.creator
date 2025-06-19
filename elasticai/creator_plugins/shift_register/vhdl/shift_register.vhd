library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.counter_pkg.all;

entity shift_register is
    generic (
        DATA_WIDTH: positive; -- size of single data point
        NUM_POINTS: positive  -- number of data points to write in a single step
    );
    port (
        clk : in std_logic;
        d_in : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        d_out : out std_logic_vector(DATA_WIDTH*NUM_POINTS - 1 downto 0);
        valid_in : in std_logic;  -- set to '1' while to write data to the register
        rst : in std_logic;  -- setting to '1' will reset the internal counter and set valid_out to `0`
        valid_out : out std_logic := '0' -- will be '1' when the register is full
    );
end entity;

architecture rtl of shift_register is
  signal q_i : std_logic_vector(d_out'range);
  signal count : natural range d_out'range;
begin


  d_out <= q_i;

  process (count) is begin
    if count = d_out'high then
      valid_out <= '1';
    else
      valid_out <= '0';
    end if;
  end process;

  process (clk, rst) is
  begin
    if rst = '1' then
      count <= d_out'low;
    elsif rising_edge(clk) then
      if valid_in = '1' and count < d_out'high then
        count <= count + 1; 
      end if;
    end if;
  end process;

  process(clk) is
  begin
    if rising_edge(clk) then
      if rst = '0' and valid_in = '1' then
          q_i <= q_i(DATA_WIDTH*(NUM_POINTS-1) - 1 downto 0) & d_in;
      end if;
    end if;
  end process;

end architecture;
