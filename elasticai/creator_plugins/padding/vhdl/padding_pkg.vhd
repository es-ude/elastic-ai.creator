library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;


package padding_pkg is
  function size_in_bytes(size: integer) return integer;

  type byte_array_t is array(integer range <>) of std_logic_vector(7 downto 0);

  function padded_size(size: integer) return integer;
  function padding_bits(size: integer) return integer;
  function clog2(n: integer) return positive;

  procedure read_from_padding_component(
    variable data_id : inout integer;
    signal rst : in std_logic;
    signal data_buffer: out std_logic_vector;
    signal d_out : in std_logic_vector;
    signal valid_out : in std_logic;
    constant DATA_DEPTH : in positive;
    constant DATA_WIDTH : in positive
  );
end package;


package body padding_pkg is
  function clog2(n: integer) return positive is
  begin
    return positive(ceil(log2(real(n))));
  end function;

  function size_in_bytes(size: integer) return integer is
    begin
      return integer(ceil(real(size) / 8.0));
  end function;

  function padded_size(size: integer) return integer is
    begin
      return size_in_bytes(size) * 8;
  end function;

  function padding_bits(size: integer) return integer is
    begin
      return padded_size(size) - size;
  end function;

  procedure read_from_padding_component(
    variable data_id : inout integer;
    signal rst : in std_logic;
    signal data_buffer: out std_logic_vector;
    signal d_out : in std_logic_vector;
    signal valid_out : in std_logic;
    constant DATA_DEPTH : in positive;
    constant DATA_WIDTH : in positive
  ) is
  begin
      if rst = '1' then
        data_id := 0;
      end if;
      if valid_out = '1' then
        if data_id < DATA_DEPTH then
          data_buffer(DATA_WIDTH*(data_id+1) - 1 downto DATA_WIDTH*data_id) <= d_out;
          data_id := data_id + 1;
        end if;
      end if;
  end procedure;

end package body;
