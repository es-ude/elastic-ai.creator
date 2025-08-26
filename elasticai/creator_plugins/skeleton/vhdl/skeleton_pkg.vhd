library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.MATH_REAL.all;

package skeleton_pkg is
  type skeleton_id_t is array (0 to 15) of std_logic_vector(7 downto 0);
  constant SKELETON_ID : skeleton_id_t := (others => x"00");

  function fmax(L: integer; R: integer) return integer;
  function fmin(L: integer; R: integer) return integer;
  function log2(val: integer) return natural;
  function get_width_in_bytes(val: integer) return integer;
end package;

package body skeleton_pkg is

  function fmax(L: integer; R : INTEGER) return INTEGER is
  begin
    if L > R then
      return L;
    else
      return R;
    end if;
  end;

  function fmin(L: integer; R : INTEGER) return INTEGER is
  begin
    if L < R then
      return L;
    else
      return R;
    end if;
  end;

  function log2(val : INTEGER) return natural is
    variable res : natural;
  begin
    for i in 0 to 31 loop
      if (val <= (2 ** i)) then
        res := i;
        exit;
      end if;
    end loop;
    return res;
  end function Log2;

  function get_width_in_bytes(val: integer) return integer is
  begin
    return integer(ceil(real(val) / real(8)));
  end function;

end package body;

