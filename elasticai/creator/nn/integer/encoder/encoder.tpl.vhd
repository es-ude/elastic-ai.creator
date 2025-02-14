library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        X_ADDR_WIDTH : integer := ${x_addr_width};
        Y_ADDR_WIDTH : integer := ${y_addr_width};
        DATA_WIDTH : integer := ${data_width};
        NUM_DIMENSIONS : integer := ${num_dimensions};
        IN_FEATURES : integer := ${in_features};
        OUT_FEATURES : integer := ${out_features}
    );
    port(
        enable: in std_logic;
        clock: in std_logic;
        x_address: out std_logic_vector(X_ADDR_WIDTH-1 downto 0);
        y_address: in std_logic_vector(Y_ADDR_WIDTH-1 downto 0);
        x: in std_logic_vector(DATA_WIDTH-1 downto 0);
        y: out std_logic_vector(DATA_WIDTH-1 downto 0);
        done: out std_logic
    );
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
    signal enc_layer_enable : std_logic;
    signal enc_layer_clock : std_logic;
    signal enc_layer_x_address : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal enc_layer_y_address : std_logic_vector(Y_ADDR_WIDTH-1 downto 0);
    signal enc_layer_x : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal enc_layer_y : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal enc_layer_done : std_logic;
begin
    enc_layer_enable <= enable;
    enc_layer_clock <= clock;
    x_address <= enc_layer_x_address;
    enc_layer_y_address <= y_address;
    enc_layer_x <= x;
    y <= enc_layer_y;
    done <= enc_layer_done;
    inst_${name}_enc_layer: entity ${work_library_name}.${name}_enc_layer(rtl)
    port map (
        enable => enc_layer_enable,
        clock => enc_layer_clock,
        x_address => enc_layer_x_address,
        y_address => enc_layer_y_address,
        x => enc_layer_x_in,
        y => enc_layer_y_out,
        done => enc_layer_done
    );
end architecture;
