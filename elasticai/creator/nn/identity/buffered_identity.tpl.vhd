library ieee;
use ieee.std_logic_1164.all;


entity $name is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x_address : out std_logic_vector($x_address_width-1 downto 0);
        y_address : in std_logic_vector($y_address_width-1 downto 0);
        x   : in std_logic_vector($x_width-1 downto 0);
        y  : out std_logic_vector($y_width-1 downto 0);
        done   : out std_logic
    );
end $name;

architecture rtl of $name is
begin
    y <= x;
    done <= enable;
    x_address <= y_address;
end rtl;
