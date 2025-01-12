library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        DATA_WIDTH : integer := ${data_width};
        CONV1DBN_0_X_ADDR_WIDTH : integer := ${conv1dbn_0_x_addr_width};
        CONV1DBN_0_Y_ADDR_WIDTH : integer := ${conv1dbn_0_y_addr_width};
        CONV1DBN_0_RELU_X_ADDR_WIDTH : integer := ${conv1dbn_0_relu_x_addr_width};
        CONV1DBN_0_RELU_Y_ADDR_WIDTH : integer := ${conv1dbn_0_relu_y_addr_width};
        CONV1DBN_1_X_ADDR_WIDTH : integer := ${conv1dbn_1_x_addr_width};
        CONV1DBN_1_Y_ADDR_WIDTH : integer := ${conv1dbn_1_y_addr_width};
        SHORTCUT_X_ADDR_WIDTH : integer := ${shortcut_x_addr_width};
        SHORTCUT_Y_ADDR_WIDTH : integer := ${shortcut_y_addr_width};
        ADD_X_ADDR_WIDTH : integer := ${add_x_addr_width};
        ADD_Y_ADDR_WIDTH : integer := ${add_y_addr_width}
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
    signal conv1dbn_0_enable : std_logic;
    signal conv1dbn_0_clock : std_logic;
    signal conv1dbn_0_x_addr : std_logic_vector(CONV1DBN_0_X_ADDR_WIDTH - 1 downto 0);
    signal conv1dbn_0_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal conv1dbn_0_y_addr : std_logic_vector(CONV1DBN_0_Y_ADDR_WIDTH - 1 downto 0);
    signal conv1dbn_0_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal conv1dbn_0_done : std_logic;
    signal conv1dbn_0_relu_enable : std_logic;
    signal conv1dbn_0_relu_clock : std_logic;
    signal conv1dbn_0_relu_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal conv1dbn_0_relu_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal conv1dbn_1_enable : std_logic;
    signal conv1dbn_1_clock : std_logic;
    signal conv1dbn_1_x_addr : std_logic_vector(CONV1DBN_1_X_ADDR_WIDTH - 1 downto 0);
    signal conv1dbn_1_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal conv1dbn_1_y_addr : std_logic_vector(CONV1DBN_1_Y_ADDR_WIDTH - 1 downto 0);
    signal conv1dbn_1_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal conv1dbn_1_done : std_logic;
    signal shortcut_enable : std_logic;
    signal shortcut_clock : std_logic;
    signal shortcut_x_addr : std_logic_vector(SHORTCUT_X_ADDR_WIDTH - 1 downto 0);
    signal shortcut_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal shortcut_y_addr : std_logic_vector(SHORTCUT_Y_ADDR_WIDTH - 1 downto 0);
    signal shortcut_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal shortcut_done : std_logic;
    signal add_enable : std_logic;
    signal add_clock : std_logic;
    signal add_x_1_addr : std_logic_vector(ADD_X_ADDR_WIDTH - 1 downto 0);
    signal add_x_1 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal add_x_2_addr : std_logic_vector(ADD_X_ADDR_WIDTH - 1 downto 0);
    signal add_x_2 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal add_y_addr : std_logic_vector(ADD_Y_ADDR_WIDTH - 1 downto 0);
    signal add_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal add_done : std_logic;
    signal relu_enable : std_logic;
    signal relu_clock : std_logic;
    signal relu_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal relu_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal relu_done : std_logic;
    begin

    conv1dbn_0_enable <= enable;
    shortcut_enable <= conv1dbn_0_done;
    x_address <= shortcut_x_addr when conv1dbn_0_done='1' else conv1dbn_0_x_addr;
    conv1dbn_0_clock <= clock;
    conv1dbn_0_x <= x;
    inst_${name}_conv1dbn_0: entity ${work_library_name}.${name}_conv1dbn_0(rtl)
    port map (
        enable => conv1dbn_0_enable,
        clock  => conv1dbn_0_clock,
        x_address  => conv1dbn_0_x_addr,
        y_address  => conv1dbn_0_y_addr,
        x  => conv1dbn_0_x,
        y => conv1dbn_0_y,
        done  => conv1dbn_0_done
    );
    conv1dbn_0_y_addr <= conv1dbn_0_relu_x_addr;
    conv1dbn_0_relu_x <= conv1dbn_0_y;
    conv1dbn_0_relu_enable <= conv1dbn_0_done;

    shortcut_clock <= clock;
    inst_${name}_shortcut_: entity ${work_library_name}.${name}_shortcut_(rtl)
    port map (
        enable => shortcut_enable,
        clock  => shortcut_clock,
        x_address  => shortcut_x_addr,
        y_address  => shortcut_y_addr,
        x  => shortcut_x,
        y => shortcut_y,
        done  => shortcut_done
    );
    shortcut_y_addr <= add_x_1_address;
    add_x_1 <= shortcut_y;
    shortcut_x <= x;

    conv1dbn_0_relu_clock <= clock;
    inst_${name}_conv1dbn_0_relu: entity ${work_library_name}.${name}_conv1dbn_0_relu(rtl)
    port map (
        enable => conv1dbn_0_relu_enable,
        clock  => conv1dbn_0_relu_clock,
        x_address  => conv1dbn_0_relu_x_addr,
        y_address  => conv1dbn_0_relu_y_addr,
        x  => conv1dbn_0_relu_x,
        y => conv1dbn_0_relu_y,
        done  => conv1dbn_0_relu_done
    );
    conv1dbn_0_relu_y_addr <= conv1dbn_1_x_addr;
    conv1dbn_1_x <= conv1dbn_0_relu_y;
    conv1dbn_1_enable <= conv1dbn_0_relu_done;

    conv1dbn_1_clock <= clock;
    inst_${name}_conv1dbn_1: entity ${work_library_name}.${name}_conv1dbn_1(rtl)
    port map (
        enable => conv1dbn_1_enable,
        clock  => conv1dbn_1_clock,
        x_address  => conv1dbn_1_x_addr,
        y_address  => conv1dbn_1_y_addr,
        x  => conv1dbn_1_x,
        y  => conv1dbn_1_y,
        done  => conv1dbn_1_done
    );
    conv1dbn_1_y_addr <= add_x_2_address;
    shortcut_y_addr <= add_x_1_address;

    add_enable <= conv1dbn_1_done and shortcut_done;
    add_clock <= clock;
    add_x_1 <= shortcut_y;
    add_x_2 <= conv1dbn_1_y;
    inst_${name}_add: entity ${work_library_name}.${name}_add(rtl)
    port map (
        enable => add_enable,
        clock  => add_clock,
        x_1_address  => add_x_1_address,
        x_1  => add_x_1,
        x_2_address  => add_x_2_address,
        x_2  => add_x_2,
        y_address  => add_y_addr,
        y => add_y,
        done  => add_done
    );
    add_y_addr <= y_address;
    relu_x <= add_y;

    relu_enable <= add_done;
    relu_clock <= clock;
    inst_${name}_relu: entity ${work_library_name}.${name}_relu(rtl)
    port map (
        enable => relu_enable,
        clock  => relu_clock,
        x  => relu_x,
        y  => relu_y
    );
    y <= relu_y;
    done <= add_done;
end architecture;
