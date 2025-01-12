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
    signal depthconv1d_0_x_addr : std_logic_vector(DEPTHCONV1D_0_X_ADDR_WIDTH - 1 downto 0);
    signal depthconv1d_0_x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal depthconv1d_0_y_addr : std_logic_vector(DEPTHCONV1D_0_Y_ADDR_WIDTH - 1 downto 0);
    signal depthconv1d_0_y_out: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal depthconv1d_0_done : std_logic;
    signal pointconv1dbn_0_enable : std_logic;
    signal pointconv1dbn_0_clock : std_logic;
    signal pointconv1dbn_0_x_addr : std_logic_vector(POINTCONV1DBN_0_X_ADDR_WIDTH - 1 downto 0);
    signal pointconv1dbn_0_x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal pointconv1dbn_0_y_addr : std_logic_vector(POINTCONV1DBN_0_Y_ADDR_WIDTH - 1 downto 0);
    signal pointconv1dbn_0_y_out: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal pointconv1dbn_0_done : std_logic;
    signal pointconv1dbn_0_relu_enable : std_logic;
    signal pointconv1dbn_0_relu_clock : std_logic;
    signal pointconv1dbn_0_relu_x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal pointconv1dbn_0_relu_y_out : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal pointconv1dbn_0_relu_done : std_logic;
    signal depthconv1d_1_enable : std_logic;
    signal depthconv1d_1_clock : std_logic;
    signal depthconv1d_1_x_addr : std_logic_vector(DEPTHCONV1D_1_X_ADDR_WIDTH - 1 downto 0);
    signal depthconv1d_1_x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal depthconv1d_1_y_addr : std_logic_vector(DEPTHCONV1D_1_Y_ADDR_WIDTH - 1 downto 0);
    signal depthconv1d_1_y_out: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal depthconv1d_1_done : std_logic;
    signal pointconv1dbn_1_enable : std_logic;
    signal pointconv1dbn_1_clock : std_logic;
    signal pointconv1dbn_1_x_addr : std_logic_vector(POINTCONV1DBN_1_X_ADDR_WIDTH - 1 downto 0);
    signal pointconv1dbn_1_x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal pointconv1dbn_1_y_addr : std_logic_vector(POINTCONV1DBN_1_Y_ADDR_WIDTH - 1 downto 0);
    signal pointconv1dbn_1_y_out: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal pointconv1dbn_1_done : std_logic;
    signal shortcut_conv1d_enable : std_logic;
    signal shortcut_conv1d_clock : std_logic;
    signal shortcut_conv1d_x_addr : std_logic_vector(SHORTCUT_CONV1D_X_ADDR_WIDTH - 1 downto 0);
    signal shortcut_conv1d_x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal shortcut_conv1d_y_addr : std_logic_vector(SHORTCUT_CONV1D_Y_ADDR_WIDTH - 1 downto 0);
    signal shortcut_conv1d_y_out : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal shortcut_conv1d_done : std_logic;
    signal add_enable : std_logic;
    signal add_clock : std_logic;
    signal add_x_1_address : std_logic_vector(ADD_X_ADDR_WIDTH - 1 downto 0);
    signal add_x_2_address : std_logic_vector(ADD_X_ADDR_WIDTH - 1 downto 0);
    signal add_x_1 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal add_x_2 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal add_y_addr : std_logic_vector(ADD_Y_ADDR_WIDTH - 1 downto 0);
    signal add_y_out : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal add_done : std_logic;
    signal relu_enable : std_logic;
    signal relu_clock : std_logic;
    signal relu_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal relu_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal relu_done : std_logic;
    begin
    depthconv1d_0_enable <= enable;
    shortcut_conv1d_enable <= depthconv1d_0_done;
    x_address <= shortcut_conv1d_x_addr when depthconv1d_0_done='1' else depthconv1d_0_x_addr;
    depthconv1d_0_clock <= clock;
    depthconv1d_0_x_in <= x;
    inst_${name}_depthconv1d_0: entity ${work_library_name}.${name}_depthconv1d_0(rtl)
    port map (
        enable => depthconv1d_0_enable,
        clock  => depthconv1d_0_clock,
        x_address  => depthconv1d_0_x_addr,
        y_address  => depthconv1d_0_y_addr,
        x  => depthconv1d_0_x_in,
        y => depthconv1d_0_y_out,
        done  => depthconv1d_0_done
    );
    depthconv1d_0_y_addr <= pointconv1dbn_0_x_addr;
    pointconv1dbn_0_x_in <= depthconv1d_0_y_out;
    pointconv1dbn_0_enable <= depthconv1d_0_done;

    shortcut_conv1d_clock <= clock;
    inst_${name}_shortcut_conv1d: entity ${work_library_name}.${name}_shortcut_conv1d(rtl)
    port map (
        enable => shortcut_conv1d_enable,
        clock  => shortcut_conv1d_clock,
        x_address  => shortcut_conv1d_x_addr,
        y_address  => shortcut_conv1d_y_addr,
        x  => shortcut_conv1d_x_in,
        y => shortcut_conv1d_y_out,
        done  => shortcut_conv1d_done
    );
    shortcut_conv1d_y_addr <= add_x_1_address;
    add_x_1 <= shortcut_conv1d_y_out;
    shortcut_conv1d_x_in <= x;

    pointconv1dbn_0_clock <= clock;
    inst_${name}_pointconv1dbn_0: entity ${work_library_name}.${name}_pointconv1dbn_0(rtl)
    port map (
        enable => pointconv1dbn_0_enable,
        clock  => pointconv1dbn_0_clock,
        x_address  => pointconv1dbn_0_x_addr,
        y_address  => pointconv1dbn_0_y_addr,
        x  => pointconv1dbn_0_x_in,
        y => pointconv1dbn_0_y_out,
        done  => pointconv1dbn_0_done
    );
    pointconv1dbn_0_y_addr <= depthconv1d_1_x_addr;
    pointconv1dbn_0_relu_x_in <= pointconv1dbn_0_y_out;
    pointconv1dbn_0_relu_enable <= pointconv1dbn_0_done;

    pointconv1dbn_0_relu_clock <= clock;
    inst_${name}_pointconv1dbn_0_relu: entity ${work_library_name}.${name}_pointconv1dbn_0_relu(rtl)
    port map (
        enable => pointconv1dbn_0_relu_enable,
        clock  => pointconv1dbn_0_relu_clock,
        x  => pointconv1dbn_0_relu_x_in,
        y  => pointconv1dbn_0_relu_y_out
    );

    depthconv1d_1_x_in <= pointconv1dbn_0_relu_y_out;

    depthconv1d_1_clock <= clock;
    depthconv1d_1_enable <= pointconv1dbn_0_done;
    inst_${name}_depthconv1d_1: entity ${work_library_name}.${name}_depthconv1d_1(rtl)
    port map (
        enable => depthconv1d_1_enable,
        clock  => depthconv1d_1_clock,
        x_address  => depthconv1d_1_x_addr,
        y_address  => depthconv1d_1_y_addr,
        x  => depthconv1d_1_x_in,
        y => depthconv1d_1_y_out,
        done  => depthconv1d_1_done
    );

    depthconv1d_1_y_addr <= pointconv1dbn_1_x_addr;
    pointconv1dbn_1_x_in <= depthconv1d_1_y_out;
    pointconv1dbn_1_enable <= depthconv1d_1_done;

    pointconv1dbn_1_clock <= clock;
    inst_${name}_pointconv1dbn_1: entity ${work_library_name}.${name}_pointconv1dbn_1(rtl)
    port map (
        enable => pointconv1dbn_1_enable,
        clock  => pointconv1dbn_1_clock,
        x_address  => pointconv1dbn_1_x_addr,
        y_address  => pointconv1dbn_1_y_addr,
        x  => pointconv1dbn_1_x_in,
        y => pointconv1dbn_1_y_out,
        done  => pointconv1dbn_1_done
    );

    pointconv1dbn_1_y_addr <= add_x_2_address;
    add_x_2 <= pointconv1dbn_1_y_out;


    add_enable <= pointconv1dbn_1_done and shortcut_conv1d_done;
    add_clock <= clock;
    inst_${name}_add: entity ${work_library_name}.${name}_add(rtl)
    port map (
        enable => add_enable,
        clock  => add_clock,
        x_1_address  => add_x_1_address,
        x_1  => add_x_1,
        x_2_address  => add_x_2_address,
        x_2  => add_x_2,
        y_address  => add_y_addr,
        y => add_y_out,
        done  => add_done
    );
    add_y_addr <= y_address;
    relu_x <= add_y_out;


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
