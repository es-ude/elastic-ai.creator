library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.all;

entity fp_network is
    port (
        enable: in std_logic;
        clock: in std_logic;

        x_address: out std_logic_vector(1-1 downto 0);
        y_address: in std_logic_vector(1-1 downto 0);

        x: in std_logic_vector(16-1 downto 0);
        y: out std_logic_vector(16-1 downto 0);

        done: out std_logic
    );
end fp_network;

architecture rtl of fp_network is

    signal fp_linear_enable : std_logic := '0';
    signal fp_linear_clock : std_logic := '0';
    signal fp_linear_done : std_logic := '0';
    signal fp_linear_x : std_logic_vector(15 downto 0);
    signal fp_linear_y : std_logic_vector(15 downto 0);
    signal fp_linear_x_address : std_logic_vector(0 downto 0);
    signal fp_linear_y_address : std_logic_vector(0 downto 0);

    -- fp_hard_sigmoid
    signal fp_hard_sigmoid_enable : std_logic := '0';
    signal fp_hard_sigmoid_clock : std_logic := '0';
    signal fp_hard_sigmoid_x : std_logic_vector(15 downto 0);
    signal fp_hard_sigmoid_y : std_logic_vector(15 downto 0);

begin

    x_address <= fp_linear_x_address;
    fp_linear_x <= x;

    --------------------------------------------------------------------------------
    -- connection between layers
    --------------------------------------------------------------------------------

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
    fp_linear : entity work.fp_linear(rtl)
    port map(
        enable => fp_linear_enable,
        clock => fp_linear_clock,

        x => fp_linear_x,
        y => fp_linear_y,
        x_address => fp_linear_x_address,
        y_address => fp_linear_y_address,
        done => fp_linear_done
    );


    fp_hard_sigmoid : entity work.fp_hard_sigmoid(rtl)
    port map(
        enable => fp_hard_sigmoid_enable,
        clock => fp_hard_sigmoid_clock,
        x => fp_hard_sigmoid_x,
        y => fp_hard_sigmoid_y
    );
       --------------------------------------------------------------------------------
       -- Instantiate all layers
       --------------------------------------------------------------------------------

end rtl;
