library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.all;

entity fp_network is
    port (
        enable  : in std_logic;
        clock   : in std_logic;

        x_addr  : out std_logic_vector(X_ADDR_WIDTH-1 downto 0);
        output_address  : in std_logic_vector($output_address_width-1 downto 0);

        x_in    : in std_logic_vector(DATA_WIDTH-1 downto 0);
        y_out   : out std_logic_vector(DATA_WIDTH-1 downto 0);

        done    : out std_logic
    );
end fp_network;

architecture rtl of fp_network is

    -- Constants for now, should be move to generic map soon
    constant LINEAR_0_X_ADDR_WIDTH : integer := X_ADDR_WIDTH;
    constant LINEAR_0_Y_ADDR_WIDTH : integer := 2;
    constant LINEAR_1_X_ADDR_WIDTH : integer := 2;
    constant LINEAR_1_Y_ADDR_WIDTH : integer := Y_ADDR_WIDTH;

    --------------------------------------------------------------------------------
    -- signals
    --------------------------------------------------------------------------------
    -- fp_linear_1d_0
    signal i_fp_linear_1d_0_enable : std_logic := '0';
    signal i_fp_linear_1d_0_clock : std_logic := '0';
    signal i_fp_linear_1d_0_x_addr : std_logic_vector(LINEAR_0_X_ADDR_WIDTH-1 downto 0);
    signal i_fp_linear_1d_0_y_addr : std_logic_vector(LINEAR_0_Y_ADDR_WIDTH-1 downto 0);
    signal i_fp_linear_1d_0_x_in : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal i_fp_linear_1d_0_y_out : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal i_fp_linear_1d_0_done : std_logic := '0';


    -- fp_linear_1d_1
    signal i_fp_linear_1d_1_enable : std_logic := '0';
    signal i_fp_linear_1d_1_clock : std_logic := '0';
    signal i_fp_linear_1d_1_x_addr : std_logic_vector(LINEAR_1_X_ADDR_WIDTH-1 downto 0);
    signal i_fp_linear_1d_1_y_addr : std_logic_vector(LINEAR_1_Y_ADDR_WIDTH-1 downto 0);
    signal i_fp_linear_1d_1_x_in : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal i_fp_linear_1d_1_y_out : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal i_fp_linear_1d_1_done : std_logic := '0';

    -- i_fp_relu
    signal i_fp_relu_enable : std_logic := '0';
    signal i_fp_relu_clock : std_logic := '0';
    signal i_fp_relu_input : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal i_fp_relu_output : std_logic_vector(DATA_WIDTH-1 downto 0);

    -- i_fp_hard_sigmoid
    signal i_fp_hard_sigmoid_enable : std_logic := '0';
    signal i_fp_hard_sigmoid_clock : std_logic := '0';
    signal i_fp_hard_sigmoid_input : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal i_fp_hard_sigmoid_output : std_logic_vector(DATA_WIDTH-1 downto 0);

begin

    x_addr <= i_fp_linear_1d_0_x_addr;
    i_fp_linear_1d_0_x_in <= x_in;

    --------------------------------------------------------------------------------
    -- connection between layers
    --------------------------------------------------------------------------------
    -- fp_linear_1d_0
    i_fp_linear_1d_0_enable <= enable;
    i_fp_linear_1d_0_clock <= clock;
    i_fp_linear_1d_0_y_addr <= i_fp_linear_1d_1_x_addr;

    -- i_fp_relu
    i_fp_relu_enable <= i_fp_linear_1d_0_done;
    i_fp_relu_clock <= clock;
    i_fp_relu_input <= i_fp_linear_1d_0_y_out;

    -- fp_linear_1d_1
    i_fp_linear_1d_1_enable <= i_fp_linear_1d_0_done;
    i_fp_linear_1d_1_clock <= clock;
    i_fp_linear_1d_1_y_addr <= y_addr;
    i_fp_linear_1d_1_x_in <= i_fp_relu_output;

    -- i_fp_hard_sigmoid
    i_fp_hard_sigmoid_enable <= i_fp_linear_1d_1_done; -- only enable when the last layer is finished.
    i_fp_hard_sigmoid_clock <= clock;
    i_fp_hard_sigmoid_input <= i_fp_linear_1d_1_y_out;
    y_out <= i_fp_hard_sigmoid_output;

    -- finally
    done <= i_fp_linear_1d_1_done;
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
