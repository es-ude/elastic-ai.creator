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
        X_COUNT : integer := ${x_count};
        Y_COUNT : integer := ${y_count}
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
    signal rnn_layer_enable : std_logic;
    signal rnn_layer_clock : std_logic;
    signal rnn_layer_x_address : std_logic_vector(log2(X_COUNT)-1 downto 0);
    signal rnn_layer_x: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal rnn_layer_y_address : std_logic_vector(log2(Y_COUNT)-1 downto 0);
    signal rnn_layer_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal rnn_layer_done : std_logic;
    begin
    rnn_layer_enable <= enable;
    done <= rnn_layer_done;
    rnn_layer_clock <= clock;
    x_address <= rnn_layer_x_address;
    rnn_layer_y_address <= y_address;
    rnn_layer_x <= x;
    y <= rnn_layer_y;
    inst_${layer_name}: entity ${work_library_name}.${layer_name}(rtl)
    port map (
        enable => rnn_layer_enable,
        clock  => rnn_layer_clock,
        x_address  => rnn_layer_x_address,
        y_address  => rnn_layer_y_address,
        x  => rnn_layer_x,
        y => rnn_layer_y,
        done  => rnn_layer_done
    );
end architecture;
