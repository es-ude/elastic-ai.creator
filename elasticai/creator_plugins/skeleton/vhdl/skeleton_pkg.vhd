library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.MATH_REAL.all;
use ieee.math_real.all;


package skeleton_pkg is
  type skeleton_id_t is array (0 to 15) of std_logic_vector(7 downto 0);
  constant SKELETON_ID : skeleton_id_t := (others => x"00");

  function fmax(L: integer; R: integer) return integer;
  function fmin(L: integer; R: integer) return integer;
  function log2(val: integer) return natural;
  function get_width_in_bytes(val: integer) return integer;
  function size_in_bytes(size: integer) return integer;

  function min_fn(a: integer; b: integer) return integer;
  function max_fn(a: integer; b: integer) return integer;

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


package body skeleton_pkg is
  function size_in_bytes(size: integer) return integer is
  begin
    return integer(ceil(real(size) / 8.0));
  end function;


  function min_fn(a: integer; b: integer) return integer is
  begin
    if a < b then
      return a;
    else
      return b;
    end if;
  end function;

  function max_fn(a: integer; b: integer) return integer is
  begin
    if a > b then
      return a;
    else
      return b;
    end if;
  end function;
end package body;
