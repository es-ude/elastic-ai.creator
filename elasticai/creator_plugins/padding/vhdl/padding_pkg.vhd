library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;


package padding_pkg is
  function size_in_bytes(size: integer) return integer;

  type byte_array_t is array(integer range <>) of std_logic_vector(7 downto 0);

  function padded_size(size: integer) return integer;
  function padding_bits(size: integer) return integer;
end package;


package body padding_pkg is

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

end package body;
