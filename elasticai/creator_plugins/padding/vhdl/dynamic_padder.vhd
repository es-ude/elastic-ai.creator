library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.padding_pkg.all;

entity dynamic_padder is

  generic (
    DATA_WIDTH : positive := 1
  );

  port (
    d_in : in std_logic_vector(DATA_WIDTH - 1 downto 0);
    d_out : out std_logic_vector(8 - 1 downto 0);
    clk : in std_logic;
    ready_in : in std_logic;
    ready_out : out std_logic;
    valid_in : in std_logic;
    valid_out : out std_logic;
    rst : in std_logic
  );

end entity;

architecture rtl of dynamic_padder is
    signal current_byte_id : integer range 0 to size_in_bytes(DATA_WIDTH) - 1 := 0;
    constant last_byte_id : integer := size_in_bytes(DATA_WIDTH) - 1;
    constant zeros : std_logic_vector(size_in_bytes(DATA_WIDTH) * 8  - DATA_WIDTH - 1 downto 0) := (others => '0');
    procedure increment_and_wrap(signal n: inout integer) is begin
      if n = last_byte_id then
        n <= 0;
      else
        n <= n + 1;
      end if;
    end procedure;
begin

  valid_out <= valid_in;

  d_out <= zeros & d_in(DATA_WIDTH - (current_byte_id* 8) - 1 downto 0) when current_byte_id = last_byte_id else d_in(DATA_WIDTH - (current_byte_id)*8 - 1 downto DATA_WIDTH - (current_byte_id + 1)*8);
  ready_out <= '1' when current_byte_id = last_byte_id and ready_in = '1' else '0';


  process (clk, rst) is
  begin
    if rst = '1' then
      current_byte_id <= 0;
    elsif rising_edge(clk) then
      if valid_in = '1' and ready_in = '1' then
        increment_and_wrap(current_byte_id);
      end if;
    end if;

  end process;


end architecture;

