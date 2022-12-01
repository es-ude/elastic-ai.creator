library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.all;

entity fp_network is
    port (
        enable  : in std_logic;
        clock   : in std_logic;

        input_addr  : out std_logic_vector(5 downto 0);
        output_addr  : in std_logic_vector(3 downto 0);

        x_in    : in std_logic_vector(15 downto 0);
        y_out   : out std_logic_vector(15 downto 0);

        done    : out std_logic
    );
end fp_network;

architecture rtl of fp_network is

    signal fp_linear_1d_0_enable : std_logic := '0';
    signal fp_linear_1d_0_clock : std_logic := '0';
    signal fp_linear_1d_0_x_addr : std_logic_vector(5 downto 0);
    signal fp_linear_1d_0_y_addr : std_logic_vector(3 downto 0);
    signal fp_linear_1d_0_x_in : std_logic_vector(15 downto 0);
    signal fp_linear_1d_0_y_out : std_logic_vector(15 downto 0);
    signal fp_linear_1d_0_done : std_logic := '0';

    -- fp_hard_sigmoid
    signal fp_hard_sigmoid_enable : std_logic := '0';
    signal fp_hard_sigmoid_clock : std_logic := '0';
    signal fp_hard_sigmoid_input : std_logic_vector(15 downto 0);
    signal fp_hard_sigmoid_output : std_logic_vector(15 downto 0);

begin

    input_addr <= fp_linear_1d_0_input_addr;
    fp_linear_1d_0_input <= input;

    --------------------------------------------------------------------------------
    -- connection between layers
    --------------------------------------------------------------------------------

    i_fp_linear_1d_0_enable <= enable;
    i_fp_linear_1d_0_clock <= clock;
    i_fp_linear_1d_0_output_addr <= output_addr;

    -- i_fp_hard_sigmoid
    fp_hard_sigmoid_enable <= fp_linear_1d_1_done; -- only enable when the last layer is finished.
    fp_hard_sigmoid_clock <= clock;
    fp_hard_sigmoid_input <= fp_linear_1d_1_output;
    output <= fp_hard_sigmoid_output;

    -- finally
    done <= fp_linear_1d_1_done;
    --------------------------------------------------------------------------------
    -- connection between layers
    --------------------------------------------------------------------------------

    --------------------------------------------------------------------------------
    -- Instantiate all layers
    --------------------------------------------------------------------------------
    i_fp_linear_1d_0 : entity work.fp_linear_1d_0(rtl)
    port map(
        enable => i_fp_linear_1d_0_enable,
        clock  => i_fp_linear_1d_0_clock,
        x_addr => i_fp_linear_1d_0_x_addr,
        y_addr => i_fp_linear_1d_0_y_addr,

        x_in   => i_fp_linear_1d_0_x_in,
        y_out  => i_fp_linear_1d_0_y_out,

        done   => i_fp_linear_1d_0_done
    );


    i_fp_linear_1d_1 : entity work.fp_linear_1d_1(rtl)
    port map(
        enable => i_fp_linear_1d_1_enable,
        clock  => i_fp_linear_1d_1_clock,
        x_addr => i_fp_linear_1d_1_x_addr,
        y_addr => i_fp_linear_1d_1_y_addr,

        x_in   => i_fp_linear_1d_1_x_in,
        y_out  => i_fp_linear_1d_1_y_out,

        done   => i_fp_linear_1d_1_done
    );

    i_fp_relu : entity work.fp_relu_3(rtl)
    port map(
        enable => i_fp_relu_enable,
        clock  => i_fp_relu_clock,
        input  => i_fp_relu_input,
        output => i_fp_relu_output
    );

    i_fp_hard_sigmoid : entity work.fp_hard_sigmoid_2(rtl)
    port map(
        enable => i_fp_hard_sigmoid_enable,
        clock  => i_fp_hard_sigmoid_clock,
        input  => i_fp_hard_sigmoid_input,
        output => i_fp_hard_sigmoid_output
    );
    --------------------------------------------------------------------------------
    -- Instantiate all layers
    --------------------------------------------------------------------------------

end rtl;
