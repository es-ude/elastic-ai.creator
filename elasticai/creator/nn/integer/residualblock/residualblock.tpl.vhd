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
    ) ;
    port (
        enable : in std_logic;
        clock : in std_logic;
        x_address : out std_logic_vector(X_ADDR_WIDTH - 1 downto 0);
        x_in : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        y_address : in std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
        y_out : out std_logic_vector(DATA_WIDTH - 1 downto 0);
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
    signal conv1dbn_1_enable : std_logic;
    signal conv1dbn_1_clock : std_logic;
    signal conv1dbn_1_x_address : std_logic_vector(X_ADDR_WIDTH - 1 downto 0);
    signal conv1dbn_1_x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal conv1dbn_1_y_address : std_logic_vector(Y_ADDR_WIDTH-1 downto 0);
    signal conv1dbn_1_y_out: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal conv1dbn_1_done : std_logic;
    signal conv1dbn_1_relu_enable : std_logic;
    signal conv1dbn_1_relu_clock : std_logic;
    signal conv1dbn_1_relu_x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal conv1dbn_1_relu_y_out : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal conv1dbn_2_enable : std_logic;
    signal conv1dbn_2_clock : std_logic;
    signal conv1dbn_2_x_address : std_logic_vector(Y_ADDR_WIDTH-1 downto 0);
    signal conv1dbn_2_x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal conv1dbn_2_y_address : std_logic_vector(Y_ADDR_WIDTH-1 downto 0);
    signal conv1dbn_2_y_out: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal conv1dbn_2_done : std_logic;
    signal shortcut_enable : std_logic;
    signal shortcut_clock : std_logic;
    signal shortcut_x_address : std_logic_vector(X_ADDR_WIDTH - 1 downto 0);
    signal shortcut_x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal shortcut_y_address : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
    signal shortcut_y_out : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal shortcut_done : std_logic;
    signal add_enable : std_logic;
    signal add_clock : std_logic;
    signal add_x_1_addr : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
    signal add_x_1 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal add_x_2_addr : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
    signal add_x_2 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal add_y_addr : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
    signal add_y_out : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal add_done : std_logic;
    signal relu_enable : std_logic;
    signal relu_clock : std_logic;
    signal relu_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal relu_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal relu_done : std_logic;
    begin
    conv1dbn_1_enable <= enable;
    conv1dbn_1_relu_enable <= conv1dbn_1_done;
    conv1dbn_2_enable <= conv1dbn_1_done;
    shortcut_enable <= enable;
    add_enable <= conv1dbn_2_done and shortcut_done;
    relu_enable <= add_done;
    conv1dbn_1_clock <= clock;
    conv1dbn_1_relu_clock <= clock;
    conv1dbn_2_clock <= clock;
    shortcut_clock <= clock;
    add_clock <= clock;
    relu_clock <= clock;
    x_address <= conv1dbn_1_x_address;
    x_address <= shortcut_x_address;
    conv1dbn_1_y_address <= conv1dbn_1_relu_x_address;
    conv1dbn_1_relu_y_address <= conv1dbn_2_x_address;
    shortcut_y_address <= add_x_1_addr;
    conv1dbn_2_y_address <= add_x_2_addr;
    add_y_addr <= relu_x_address;
    relu_y_address <= y_address;
    conv1dbn_1_x_in <= x_in;
    shortcut_x_in <= x_in;
    conv1dbn_1_relu_x_in <= conv1dbn_1_y_out;
    conv1dbn_2_x_in <= conv1dbn_1_relu_y_out;
    add_x_1 <= shortcut_y_out;
    add_x_2 <= conv1dbn_2_y_out;
    relu_x <= add_y_out;
    y_out <= relu_y;
    done <= relu_done;
    inst_${name}_conv1dbn_1: ${work_library_name},${name}_conv1dbn_1(rtl)
    port map (
        enable => conv1dbn_1_enable,
        clock  => conv1dbn_1_clock,
        x_address  => conv1dbn_1_x_address,
        y_address  => conv1dbn_1_y_address,
        x_in  => conv1dbn_1_x_in,
        y_out => conv1dbn_1_y_out,
        done  => conv1dbn_1_done
    );
    inst_${name}_conv1dbn_1_relu: ${work_library_name},${name}_conv1dbn_1_relu(rtl)
    port map (
        enable => conv1dbn_1_relu_enable,
        clock  => conv1dbn_1_relu_clock,
        x  => conv1dbn_1_relu_x_in,
        y => conv1dbn_1_relu_y_out
    );
    inst_${name}_conv1dbn_2: ${work_library_name},${name}_conv1dbn_2(rtl)
    port map (
        enable => conv1dbn_2_enable,
        clock  => conv1dbn_2_clock,
        x_address  => conv1dbn_2_x_address,
        y_address  => conv1dbn_2_y_address,
        x_in  => conv1dbn_2_x_in,
        y_out => conv1dbn_2_y_out,
        done  => conv1dbn_2_done
    );
    inst_${name}_shortcut: ${work_library_name},${name}_shortcut(rtl)
    port_map (
        enable => shortcut_enable,
        clock  => shortcut_clock,
        x_address  => shortcut_x_address,
        y_address  => shortcut_y_address,
        x_in  => shortcut_x_in,
        y_out => shortcut_y_out,
        done  => shortcut_done
    )
    inst_${name}_add: ${work_library_name},${name}_add(rtl)
    port_map (
        enable => add_enable,
        clock  => add_clock,
        x_1_addr  => add_x_1_addr,
        x_1  => add_x_1,
        x_2_addr  => add_x_2_addr,
        x_2  => add_x_2,
        y_addr  => add_y_addr,
        y_out => add_y_out,
        done  => add_done
    )
    inst_${name}_relu: ${work_library_name},${name}_relu(rtl)
    port_map (
        enable => relu_enable,
        clock  => relu_clock,
        x  => relu_x,
        y  => relu_y,
        done  => relu_done
    )
end architecture;
