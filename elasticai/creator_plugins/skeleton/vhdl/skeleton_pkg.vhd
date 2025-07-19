library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.math_real.all;


package skeleton_pkg is
  type skeleton_id_t is array (0 to 15) of std_logic_vector(7 downto 0);
  constant SKELETON_ID : skeleton_id_t := (others => x"00");
  function size_in_bytes(size: integer) return integer;


end package;

package body skeleton_pkg is
  function size_in_bytes(size: integer) return integer is
  begin
    return integer(ceil(real(size) / 8.0));
  end function;


end package body;
