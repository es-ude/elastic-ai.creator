library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.all;

entity $layer_name is
    port (
        enable: in std_logic;
        clock: in std_logic;

        x_address: out std_logic_vector($x_address_width-1 downto 0);
        y_address: in std_logic_vector($y_address_width-1 downto 0);

        x: in std_logic_vector($x_width-1 downto 0);
        y: out std_logic_vector($y_width-1 downto 0);

        done: out std_logic
    );
end $layer_name;

architecture rtl of $layer_name is
    $signal_definitions
begin
    $layer_connections
    --------------------------------------------------------------------------------
    -- Instantiate all layers
    --------------------------------------------------------------------------------
    $layer_instantiations
end rtl;
