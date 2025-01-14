library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        DATA_WIDTH : integer := ${data_width};
        DEPTHCONV1D_0_X_ADDR_WIDTH : integer := ${depthconv1d_0_x_addr_width};
        DEPTHCONV1D_0_Y_ADDR_WIDTH : integer := ${depthconv1d_0_y_addr_width};
        POINTCONV1DBN_0_X_ADDR_WIDTH : integer := ${pointconv1dbn_0_x_addr_width};
        POINTCONV1DBN_0_Y_ADDR_WIDTH : integer := ${pointconv1dbn_0_y_addr_width};
        DEPTHCONV1D_1_X_ADDR_WIDTH : integer := ${depthconv1d_1_x_addr_width};
        DEPTHCONV1D_1_Y_ADDR_WIDTH : integer := ${depthconv1d_1_y_addr_width};
        POINTCONV1DBN_1_X_ADDR_WIDTH : integer := ${pointconv1dbn_1_x_addr_width};
        POINTCONV1DBN_1_Y_ADDR_WIDTH : integer := ${pointconv1dbn_1_y_addr_width};
        SHORTCUT_CONV1D_X_ADDR_WIDTH : integer := ${shortcut_conv1d_x_addr_width};
        SHORTCUT_CONV1D_Y_ADDR_WIDTH : integer := ${shortcut_conv1d_y_addr_width};
        ADD_X_ADDR_WIDTH : integer := ${add_x_addr_width};
        ADD_Y_ADDR_WIDTH : integer := ${add_y_addr_width}
    ) ;
    port (
        enable : in std_logic;
        clock : in std_logic;
        x_address : out std_logic_vector(DEPTHCONV1D_0_X_ADDR_WIDTH - 1 downto 0);
        x : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        y_address : in std_logic_vector(ADD_Y_ADDR_WIDTH - 1 downto 0);
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
    signal depthconv1d_0_enable : std_logic;
    signal depthconv1d_0_clock : std_logic;
    signal depthconv1d_0_x_address : std_logic_vector(DEPTHCONV1D_0_X_ADDR_WIDTH - 1 downto 0);
    signal depthconv1d_0_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal depthconv1d_0_y_address : std_logic_vector(DEPTHCONV1D_0_Y_ADDR_WIDTH - 1 downto 0);
    signal depthconv1d_0_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal depthconv1d_0_done : std_logic;
    signal pointconv1dbn_0_enable : std_logic;
    signal pointconv1dbn_0_clock : std_logic;
    signal pointconv1dbn_0_x_address : std_logic_vector(POINTCONV1DBN_0_X_ADDR_WIDTH - 1 downto 0);
    signal pointconv1dbn_0_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal pointconv1dbn_0_y_address : std_logic_vector(POINTCONV1DBN_0_Y_ADDR_WIDTH - 1 downto 0);
    signal pointconv1dbn_0_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal pointconv1dbn_0_done : std_logic;
    signal pointconv1dbn_0_relu_enable : std_logic;
    signal pointconv1dbn_0_relu_clock : std_logic;
    signal pointconv1dbn_0_relu_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal pointconv1dbn_0_relu_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal depthconv1d_1_enable : std_logic;
    signal depthconv1d_1_clock : std_logic;
    signal depthconv1d_1_x_address : std_logic_vector(DEPTHCONV1D_1_X_ADDR_WIDTH - 1 downto 0);
    signal depthconv1d_1_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal depthconv1d_1_y_address : std_logic_vector(DEPTHCONV1D_1_Y_ADDR_WIDTH - 1 downto 0);
    signal depthconv1d_1_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal depthconv1d_1_done : std_logic;
    signal pointconv1dbn_1_enable : std_logic;
    signal pointconv1dbn_1_clock : std_logic;
    signal pointconv1dbn_1_x_address : std_logic_vector(POINTCONV1DBN_1_X_ADDR_WIDTH - 1 downto 0);
    signal pointconv1dbn_1_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal pointconv1dbn_1_y_address : std_logic_vector(POINTCONV1DBN_1_Y_ADDR_WIDTH - 1 downto 0);
    signal pointconv1dbn_1_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal pointconv1dbn_1_done : std_logic;
    signal shortcut_conv1d_enable : std_logic;
    signal shortcut_conv1d_clock : std_logic;
    signal shortcut_conv1d_x_address : std_logic_vector(SHORTCUT_CONV1D_X_ADDR_WIDTH - 1 downto 0);
    signal shortcut_conv1d_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal shortcut_conv1d_y_address : std_logic_vector(SHORTCUT_CONV1D_Y_ADDR_WIDTH - 1 downto 0);
    signal shortcut_conv1d_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal shortcut_conv1d_done : std_logic;
    signal add_enable : std_logic;
    signal add_clock : std_logic;
    signal add_x_1_address : std_logic_vector(ADD_X_ADDR_WIDTH - 1 downto 0);
    signal add_x_2_address : std_logic_vector(ADD_X_ADDR_WIDTH - 1 downto 0);
    signal add_x_1 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal add_x_2 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal add_y_address : std_logic_vector(ADD_Y_ADDR_WIDTH - 1 downto 0);
    signal add_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal add_done : std_logic;
    signal relu_enable : std_logic;
    signal relu_clock : std_logic;
    signal relu_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal relu_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    begin
    depthconv1d_0_enable <= enable;
    shortcut_conv1d_enable <= depthconv1d_0_done;
    pointconv1dbn_0_enable <= depthconv1d_0_done;
    pointconv1dbn_0_relu_enable <= pointconv1dbn_0_done;
    depthconv1d_1_enable <= pointconv1dbn_0_done;
    pointconv1dbn_1_enable <= depthconv1d_1_done;
    add_enable <= pointconv1dbn_1_done and shortcut_conv1d_done;
    relu_enable <= add_done;
    done <= add_done;
    depthconv1d_0_clock <= clock;
    shortcut_conv1d_clock <= clock;
    pointconv1dbn_0_clock <= clock;
    pointconv1dbn_0_relu_clock <= clock;
    depthconv1d_1_clock <= clock;
    pointconv1dbn_1_clock <= clock;
    add_clock <= clock;
    relu_clock <= clock;
    x_address <= shortcut_conv1d_x_address when depthconv1d_0_done='1' else depthconv1d_0_x_address;
    depthconv1d_0_y_address <= pointconv1dbn_0_x_address;
    pointconv1dbn_0_y_address <= depthconv1d_1_x_address;
    depthconv1d_1_y_address <= pointconv1dbn_1_x_address;
    shortcut_conv1d_y_address <= add_x_1_address;
    pointconv1dbn_1_y_address <= add_x_2_address;
    add_y_address <= y_address;
    depthconv1d_0_x <= x;
    shortcut_conv1d_x <= x;
    pointconv1dbn_0_x <= depthconv1d_0_y;
    pointconv1dbn_0_relu_x <= pointconv1dbn_0_y;
    depthconv1d_1_x <= pointconv1dbn_0_relu_y;
    pointconv1dbn_1_x <= depthconv1d_1_y;
    add_x_1 <= shortcut_conv1d_y;
    add_x_2 <= pointconv1dbn_1_y;
    relu_x <= add_y;
    y <= relu_y;
    inst_${name}_depthconv1d_0: entity ${work_library_name}.${name}_depthconv1d_0(rtl)
    port map (
        enable => depthconv1d_0_enable,
        clock  => depthconv1d_0_clock,
        x_address  => depthconv1d_0_x_address,
        y_address  => depthconv1d_0_y_address,
        x  => depthconv1d_0_x,
        y => depthconv1d_0_y,
        done  => depthconv1d_0_done
    );
    inst_${name}_shortcut_conv1d: entity ${work_library_name}.${name}_shortcut_conv1d(rtl)
    port map (
        enable => shortcut_conv1d_enable,
        clock  => shortcut_conv1d_clock,
        x_address  => shortcut_conv1d_x_address,
        y_address  => shortcut_conv1d_y_address,
        x  => shortcut_conv1d_x,
        y => shortcut_conv1d_y,
        done  => shortcut_conv1d_done
    );
    inst_${name}_pointconv1dbn_0: entity ${work_library_name}.${name}_pointconv1dbn_0(rtl)
    port map (
        enable => pointconv1dbn_0_enable,
        clock  => pointconv1dbn_0_clock,
        x_address  => pointconv1dbn_0_x_address,
        y_address  => pointconv1dbn_0_y_address,
        x  => pointconv1dbn_0_x,
        y => pointconv1dbn_0_y,
        done  => pointconv1dbn_0_done
    );
    inst_${name}_pointconv1dbn_0_relu: entity ${work_library_name}.${name}_pointconv1dbn_0_relu(rtl)
    port map (
        enable => pointconv1dbn_0_relu_enable,
        clock  => pointconv1dbn_0_relu_clock,
        x  => pointconv1dbn_0_relu_x,
        y  => pointconv1dbn_0_relu_y
    );
    inst_${name}_depthconv1d_1: entity ${work_library_name}.${name}_depthconv1d_1(rtl)
    port map (
        enable => depthconv1d_1_enable,
        clock  => depthconv1d_1_clock,
        x_address  => depthconv1d_1_x_address,
        y_address  => depthconv1d_1_y_address,
        x  => depthconv1d_1_x,
        y => depthconv1d_1_y,
        done  => depthconv1d_1_done
    );
    inst_${name}_pointconv1dbn_1: entity ${work_library_name}.${name}_pointconv1dbn_1(rtl)
    port map (
        enable => pointconv1dbn_1_enable,
        clock  => pointconv1dbn_1_clock,
        x_address  => pointconv1dbn_1_x_address,
        y_address  => pointconv1dbn_1_y_address,
        x  => pointconv1dbn_1_x,
        y => pointconv1dbn_1_y,
        done  => pointconv1dbn_1_done
    );
    inst_${name}_add: entity ${work_library_name}.${name}_add(rtl)
    port map (
        enable => add_enable,
        clock  => add_clock,
        x_1_address  => add_x_1_address,
        x_1  => add_x_1,
        x_2_address  => add_x_2_address,
        x_2  => add_x_2,
        y_address  => add_y_address,
        y => add_y,
        done  => add_done
    );
    inst_${name}_relu: entity ${work_library_name}.${name}_relu(rtl)
    port map (
        enable => relu_enable,
        clock  => relu_clock,
        x  => relu_x,
        y  => relu_y
    );
end architecture;
