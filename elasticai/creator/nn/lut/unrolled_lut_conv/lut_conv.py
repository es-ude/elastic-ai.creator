class LutConv:
    def __init__(self, luts):
        pass

    def save_to(self):
        template = """library ieee;
use ieee.std_logic_1164.all;

entity $name is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x   : in std_logic_vector($x_width-1 downto 0);
        y  : out std_logic_vector($y_width-1 downto 0);
    );
end;

architecture rtl of $name is
begin
    $luts
end;
"""
