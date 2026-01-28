library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

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
  function clog2(n : natural) return natural is
  begin
    if n <= 1 then
      return 1;
    else
      return natural(ceil(log2(real(n))));
    end if;
  end function;

  signal q_i : std_logic_vector(d_out'range);
  signal point_counter : unsigned(clog2(NUM_POINTS + 1) - 1 downto 0) := (others => '0');
begin

  valid_out <= '1' when point_counter = to_unsigned(NUM_POINTS, point_counter'length)
               else '0';

  d_out <= q_i;


  process(clk, rst)
  begin
    if rst = '1' then
        q_i <= (others => '0');
        point_counter <= (others => '0');
    elsif rising_edge(clk) then
        if valid_in = '1' then
            q_i <= q_i(DATA_WIDTH*(NUM_POINTS-1) - 1 downto 0) & d_in;
            if point_counter < to_unsigned(NUM_POINTS, point_counter'length) then
                point_counter <= point_counter + 1;
            end if;
        end if;
    end if;
  end process;

end architecture;
