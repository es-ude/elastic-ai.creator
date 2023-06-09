library ieee;
use ieee.std_logic_1164.all;


entity $name is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x   : in std_logic_vector($x_width-1 downto 0);
        y  : out std_logic_vector($y_width-1 downto 0);
    );
end $name;

architecture rtl of $name is
begin
    y <= x;
end rtl;
