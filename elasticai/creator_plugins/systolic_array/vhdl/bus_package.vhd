library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

package bus_package is
    type bus_array_4_8 is array (0 to 3) of std_logic_vector(7 downto 0);
    type dim_array is array (0 to 3) of integer range 0 to 32;
end package bus_package;

package body bus_package is
end package body bus_package;
