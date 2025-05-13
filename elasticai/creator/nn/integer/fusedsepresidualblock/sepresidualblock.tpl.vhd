library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        DATA_WIDTH : integer := ${data_width};
        SEPCONV1DBN_0_X_ADDR_WIDTH : integer := ${sepconv1dbn_0_x_addr_width};
        SEPCONV1DBN_0_Y_ADDR_WIDTH : integer := ${sepconv1dbn_0_y_addr_width};
        SEPCONV1DBN_1_X_ADDR_WIDTH : integer := ${sepconv1dbn_1_x_addr_width};
        SEPCONV1DBN_1_Y_ADDR_WIDTH : integer := ${sepconv1dbn_1_y_addr_width};
        SHORTCUT_X_ADDR_WIDTH : integer := ${shortcut_x_addr_width};
        SHORTCUT_Y_ADDR_WIDTH : integer := ${shortcut_y_addr_width};
        ADD_X_ADDR_WIDTH : integer := ${add_x_addr_width};
        ADD_Y_ADDR_WIDTH : integer := ${add_y_addr_width}
    ) ;
    port (
        enable : in std_logic;
        clock : in std_logic;
        x_address : out std_logic_vector(SEPCONV1DBN_0_X_ADDR_WIDTH - 1 downto 0);
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
    signal sepconv1dbn_0_enable : std_logic;
    signal sepconv1dbn_0_clock : std_logic;
    signal sepconv1dbn_0_x_address : std_logic_vector(SEPCONV1DBN_0_X_ADDR_WIDTH - 1 downto 0);
    signal sepconv1dbn_0_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal sepconv1dbn_0_y_address : std_logic_vector(SEPCONV1DBN_0_Y_ADDR_WIDTH - 1 downto 0);
    signal sepconv1dbn_0_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal sepconv1dbn_0_done : std_logic;

    signal sepconv1dbn_0_relu_enable : std_logic;
    signal sepconv1dbn_0_relu_clock : std_logic;
    signal sepconv1dbn_0_relu_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal sepconv1dbn_0_relu_y : std_logic_vector(DATA_WIDTH - 1 downto 0);

    signal sepconv1dbn_1_enable : std_logic;
    signal sepconv1dbn_1_clock : std_logic;
    signal sepconv1dbn_1_x_address : std_logic_vector(SEPCONV1DBN_1_X_ADDR_WIDTH - 1 downto 0);
    signal sepconv1dbn_1_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal sepconv1dbn_1_y_address : std_logic_vector(SEPCONV1DBN_1_Y_ADDR_WIDTH - 1 downto 0);
    signal sepconv1dbn_1_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal sepconv1dbn_1_done : std_logic;

    signal shortcut_enable : std_logic;
    signal shortcut_clock : std_logic;
    signal shortcut_x_address : std_logic_vector(SHORTCUT_X_ADDR_WIDTH - 1 downto 0);
    signal shortcut_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal shortcut_y_address : std_logic_vector(SHORTCUT_Y_ADDR_WIDTH - 1 downto 0);
    signal shortcut_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal shortcut_done : std_logic;
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
    sepconv1dbn_0_enable <= enable;
    shortcut_enable <= sepconv1dbn_0_done;
    sepconv1dbn_0_relu_enable <= sepconv1dbn_0_done;
    sepconv1dbn_1_enable <= sepconv1dbn_0_done;
    add_enable <= sepconv1dbn_1_done and shortcut_done;
    relu_enable <= add_done;
    done <= add_done;

    sepconv1dbn_0_clock <= clock;
    shortcut_clock <= clock;
    sepconv1dbn_0_relu_clock <= clock;
    sepconv1dbn_1_clock <= clock;
    add_clock <= clock;
    relu_clock <= clock;

    x_address <= shortcut_x_address when sepconv1dbn_0_done='1' else sepconv1dbn_0_x_address;
    sepconv1dbn_0_y_address <= sepconv1dbn_1_x_address;
    shortcut_y_address <= add_x_1_address;
    sepconv1dbn_1_y_address <= add_x_2_address;
    add_y_address <= y_address;

    sepconv1dbn_0_x <= x;
    shortcut_x <= x;
    sepconv1dbn_0_relu_x <= sepconv1dbn_0_y;
    sepconv1dbn_0_x <= sepconv1dbn_0_relu_y;
    add_x_1 <= shortcut_y;
    add_x_2 <= sepconv1dbn_1_y;
    relu_x <= add_y;
    y <= relu_y;
    inst_${name}_sepconv1dbn_0: entity ${work_library_name}.${name}_sepconv1dbn_0(rtl)
    port map (
        enable => sepconv1dbn_0_enable,
        clock  => sepconv1dbn_0_clock,
        x_address  => sepconv1dbn_0_x_address,
        y_address  => sepconv1dbn_0_y_address,
        x  => sepconv1dbn_0_x,
        y => sepconv1dbn_0_y,
        done  => sepconv1dbn_0_done
    );
    inst_${name}_shortcut: entity ${work_library_name}.${name}_shortcut(rtl)
    port map (
        enable => shortcut_enable,
        clock  => shortcut_clock,
        x_address  => shortcut_x_address,
        y_address  => shortcut_y_address,
        x  => shortcut_x,
        y => shortcut_y,
        done  => shortcut_done
    );
    inst_${name}_sepconv1dbn_0_relu: entity ${work_library_name}.${name}_sepconv1dbn_0_relu(rtl)
    port map (
        enable => sepconv1dbn_0_relu_enable,
        clock  => sepconv1dbn_0_relu_clock,
        x  => sepconv1dbn_0_relu_x,
        y  => sepconv1dbn_0_relu_y,
    );
    inst_${name}_sepconv1dbn_1: entity ${work_library_name}.${name}_sepconv1dbn_1(rtl)
    port map (
        enable => sepconv1dbn_1_enable,
        clock  => sepconv1dbn_1_clock,
        x_address  => sepconv1dbn_1_x_address,
        y_address  => sepconv1dbn_1_y_address,
        x  => sepconv1dbn_1_x,
        y => sepconv1dbn_1_y,
        done  => sepconv1dbn_1_done
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
