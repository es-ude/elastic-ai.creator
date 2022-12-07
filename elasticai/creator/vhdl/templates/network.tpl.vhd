library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.all;

entity fp_network is
    port (
        enable: in std_logic;
        clock: in std_logic;

        x_address: out std_logic_vector($x_address_width-1 downto 0);
        y_address: in std_logic_vector($y_address_width-1 downto 0);

        x: in std_logic_vector($data_width-1 downto 0);
        y: out std_logic_vector($data_width-1 downto 0);

        done: out std_logic
    );
end fp_network;

architecture rtl of fp_network is
    $signal_definitions
begin

    x_address <= fp_linear_x_address;
    fp_linear_x <= x;

    --------------------------------------------------------------------------------
    -- connection between layers
    --------------------------------------------------------------------------------
    -- fp_linear
    fp_linear_enable <= enable;
    fp_linear_clock <= clock;
    fp_linear_y_address <= y_address;

    -- fp_hard_sigmoid
    fp_hard_sigmoid_enable <= fp_linear_done; -- only enable when the last layer is finished.
    fp_hard_sigmoid_clock <= clock;
    fp_hard_sigmoid_x <= fp_linear_y;
    y <= fp_hard_sigmoid_y;

    -- finally
    done <= fp_linear_done;
    --------------------------------------------------------------------------------
    -- connection between layers
    --------------------------------------------------------------------------------

    --------------------------------------------------------------------------------
    -- Instantiate all layers
    --------------------------------------------------------------------------------
    $layer_instantiations
    --------------------------------------------------------------------------------
    -- Instantiate all layers
    --------------------------------------------------------------------------------

end rtl;
