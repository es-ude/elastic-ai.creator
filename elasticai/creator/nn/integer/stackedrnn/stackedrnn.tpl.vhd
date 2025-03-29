library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        DATA_WIDTH : integer := ${data_width};
        X_ADDR_WIDTH : integer := ${x_addr_width};
        Y_ADDR_WIDTH : integer := ${y_addr_width};
        X_1_COUNT : integer := ${x_1_count};
        X_2_COUNT : integer := ${x_2_count};
        X_3_COUNT : integer := ${x_3_count};
        Y_1_COUNT : integer := ${y_1_count};
        Y_2_COUNT : integer := ${y_2_count};
        Y_3_COUNT : integer := ${y_3_count}
    ) ;
    port (
        enable : in std_logic;
        clock : in std_logic;
        x_address : out std_logic_vector(X_ADDR_WIDTH - 1 downto 0);
        x : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        y_address : in std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
        y : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        done : out std_logic
    ) ;
end ${name};
architecture rtl of ${name} is
    function log2(val : INTEGER) return natural is
        variable result : natural;
    begin
        for i in 1 to 31 loop
            if (val <= (2 ** i)) then
                result := i;
                exit;
            end if;
        end loop;
        return result;
    end function log2;
    signal rnn_layer_0_enable : std_logic;
    signal rnn_layer_0_clock : std_logic;
    signal rnn_layer_0_x1_address : std_logic_vector(log2(X_1_COUNT)-1 downto 0);
    signal rnn_layer_0_x2_address : std_logic_vector(log2(X_2_COUNT)-1 downto 0);
    signal rnn_layer_0_x3_address : std_logic_vector(log2(X_3_COUNT)-1 downto 0);
    signal rnn_layer_0_y1_address : std_logic_vector(log2(Y_1_COUNT)-1 downto 0);
    signal rnn_layer_0_y2_address : std_logic_vector(log2(Y_2_COUNT)-1 downto 0);
    signal rnn_layer_0_y3_address : std_logic_vector(log2(Y_3_COUNT)-1 downto 0);
    signal rnn_layer_0_x1: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal rnn_layer_0_x2: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal rnn_layer_0_x3: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal rnn_layer_0_y1: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal rnn_layer_0_y2: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal rnn_layer_0_y3: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal rnn_layer_0_done : std_logic;
    begin
    rnn_layer_0_enable <= enable;
    rnn_layer_0_clock <= clock;
    x_address <= rnn_layer_0_x1_address;
    rnn_layer_0_x1 <= x;
    rnn_layer_0_y2_address <= y_address;
    y <= rnn_layer_0_y2;
    done <= rnn_layer_0_done;

    -- we don't need to use them, since the x2 and x3 will be always 0 for the first layer
    -- rnn_layer_0_x2_address leave to open
    -- rnn_layer_0_x3_address leave to open
    rnn_layer_0_x2 <= (others => '0');
    rnn_layer_0_x3 <= (others => '0');

    -- the stacked_rnn_0 doesn't provide information about the y1 and y3,
    -- so we always assign their address to 0
    -- rnn_layer_0_y1_address connect to zero
    -- rnn_layer_0_y3_address lconnect to zero
    rnn_layer_0_y1_address <= (others => '0');
    rnn_layer_0_y3_address <= (others => '0');
    -- y1 and y3 are left to open

    inst_${layer_name}: entity ${work_library_name}.${layer_name}(rtl)
    port map (
        enable => rnn_layer_0_enable,
        clock  => rnn_layer_0_clock,
        x_1_address => rnn_layer_0_x1_address, -- input data
        x_2_address => rnn_layer_0_x2_address, -- hidden state
        x_3_address => rnn_layer_0_x3_address, -- cell state
        y_1_address => rnn_layer_0_y1_address, -- hidden states
        y_2_address => rnn_layer_0_y2_address, -- hidden state
        y_3_address => rnn_layer_0_y3_address, -- cell state
        x_1  => rnn_layer_0_x1,
        x_2  => rnn_layer_0_x2,
        x_3  => rnn_layer_0_x3,
        y_1  => rnn_layer_0_y1,
        y_2  => rnn_layer_0_y2,
        y_3  => rnn_layer_0_y3,
        done  => rnn_layer_0_done
    );
end architecture;
