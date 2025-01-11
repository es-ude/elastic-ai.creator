library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        DATA_WIDTH : integer := ${data_width};
        DEPTHWISE_CONV1D_0_X_ADDR_WIDTH : integer := ${depthwise_conv1d_0_x_addr_width};
        DEPTHWISE_CONV1D_0_Y_ADDR_WIDTH : integer := ${depthwise_conv1d_0_y_addr_width};
        POINTWISE_CONV1DBN_0_X_ADDR_WIDTH : integer := ${pointwise_conv1dbn_0_x_addr_width};
        POINTWISE_CONV1DBN_0_Y_ADDR_WIDTH : integer := ${pointwise_conv1dbn_0_y_addr_width};
        DEPTHWISE_CONV1D_1_X_ADDR_WIDTH : integer := ${depthwise_conv1d_1_x_addr_width};
        DEPTHWISE_CONV1D_1_Y_ADDR_WIDTH : integer := ${depthwise_conv1d_1_y_addr_width};
        POINTWISE_CONV1DBN_1_X_ADDR_WIDTH : integer := ${pointwise_conv1dbn_1_x_addr_width};
        POINTWISE_CONV1DBN_1_Y_ADDR_WIDTH : integer := ${pointwise_conv1dbn_1_y_addr_width};
        SHORTCUT_CONV1D_X_ADDR_WIDTH : integer := ${shortcut_conv1d_x_addr_width};
        SHORTCUT_CONV1D_Y_ADDR_WIDTH : integer := ${shortcut_conv1d_y_addr_width};
        ADD_X_ADDR_WIDTH : integer := ${add_x_addr_width};
        ADD_Y_ADDR_WIDTH : integer := ${add_y_addr_width}
    ) ;
    port (
        enable : in std_logic;
        clock : in std_logic;
        x_addr : out std_logic_vector(DEPTHWISE_CONV1D_0_X_ADDR_WIDTH - 1 downto 0);
        x_in : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        y_addr : in std_logic_vector(ADD_Y_ADDR_WIDTH - 1 downto 0);
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
    signal depthwise_conv1d_0_enable : std_logic;
    signal depthwise_conv1d_0_clock : std_logic;
    signal depthwise_conv1d_0_x_addr : std_logic_vector(DEPTHWISE_CONV1D_0_X_ADDR_WIDTH - 1 downto 0);
    signal depthwise_conv1d_0_x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal depthwise_conv1d_0_y_addr : std_logic_vector(DEPTHWISE_CONV1D_0_Y_ADDR_WIDTH - 1 downto 0);
    signal depthwise_conv1d_0_y_out: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal depthwise_conv1d_0_done : std_logic;
    signal pointwise_conv1dbn_0_enable : std_logic;
    signal pointwise_conv1dbn_0_clock : std_logic;
    signal pointwise_conv1dbn_0_x_addr : std_logic_vector(POINTWISE_CONV1DBN_0_X_ADDR_WIDTH - 1 downto 0);
    signal pointwise_conv1dbn_0_x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal pointwise_conv1dbn_0_y_addr : std_logic_vector(POINTWISE_CONV1DBN_0_Y_ADDR_WIDTH - 1 downto 0);
    signal pointwise_conv1dbn_0_y_out: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal pointwise_conv1dbn_0_done : std_logic;
    signal pointwise_conv1dbn_0_relu_enable : std_logic;
    signal pointwise_conv1dbn_0_relu_clock : std_logic;
    signal pointwise_conv1dbn_0_relu_x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal pointwise_conv1dbn_0_relu_y_out : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal pointwise_conv1dbn_0_relu_done : std_logic;
    signal depthwise_conv1d_1_enable : std_logic;
    signal depthwise_conv1d_1_clock : std_logic;
    signal depthwise_conv1d_1_x_addr : std_logic_vector(DEPTHWISE_CONV1D_1_X_ADDR_WIDTH - 1 downto 0);
    signal depthwise_conv1d_1_x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal depthwise_conv1d_1_y_addr : std_logic_vector(DEPTHWISE_CONV1D_1_Y_ADDR_WIDTH - 1 downto 0);
    signal depthwise_conv1d_1_y_out: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal depthwise_conv1d_1_done : std_logic;
    signal pointwise_conv1dbn_1_enable : std_logic;
    signal pointwise_conv1dbn_1_clock : std_logic;
    signal pointwise_conv1dbn_1_x_addr : std_logic_vector(POINTWISE_CONV1DBN_1_X_ADDR_WIDTH - 1 downto 0);
    signal pointwise_conv1dbn_1_x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal pointwise_conv1dbn_1_y_addr : std_logic_vector(POINTWISE_CONV1DBN_1_Y_ADDR_WIDTH - 1 downto 0);
    signal pointwise_conv1dbn_1_y_out: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal pointwise_conv1dbn_1_done : std_logic;
    signal shortcut_conv1d_enable : std_logic;
    signal shortcut_conv1d_clock : std_logic;
    signal shortcut_conv1d_x_addr : std_logic_vector(SHORTCUT_CONV1D_X_ADDR_WIDTH - 1 downto 0);
    signal shortcut_conv1d_x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal shortcut_conv1d_y_addr : std_logic_vector(SHORTCUT_CONV1D_Y_ADDR_WIDTH - 1 downto 0);
    signal shortcut_conv1d_y_out : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal shortcut_conv1d_done : std_logic;
    signal add_enable : std_logic;
    signal add_clock : std_logic;
    signal add_x_1_addr : std_logic_vector(ADD_X_ADDR_WIDTH - 1 downto 0);
    signal add_x_2_addr : std_logic_vector(ADD_X_ADDR_WIDTH - 1 downto 0);
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

    --depthwise_conv1d_0 - > pointwise_conv1dbn_0 - > pointwise_conv1dbn_0_relu - > depthwise_conv1d_1 - > pointwise_conv1dbn_1 - > add_x2
    --shortcut_conv1d -> add_x1
    --- relu

    -- shared resources
    depthwise_conv1d_0_enable <= enable;
    shortcut_conv1d_enable <= depthwise_conv1d_0_done; -- enable and
    x_addr <= shortcut_conv1d_x_addr when depthwise_conv1d_0_done='1' else depthwise_conv1d_0_x_addr;
    -- shared resources

    depthwise_conv1d_0_clock <= clock;
    depthwise_conv1d_0_x_in <= x_in;
    inst_${name}_depthwise_conv1d_0: entity ${work_library_name}.${name}_depthwise_conv1d_0(rtl)
    port map (
        enable => depthwise_conv1d_0_enable,
        clock  => depthwise_conv1d_0_clock,
        x_addr  => depthwise_conv1d_0_x_addr,
        y_addr  => depthwise_conv1d_0_y_addr,
        x_in  => depthwise_conv1d_0_x_in,
        y_out => depthwise_conv1d_0_y_out,
        done  => depthwise_conv1d_0_done
    );
    depthwise_conv1d_0_y_addr <= pointwise_conv1dbn_0_x_addr;
    pointwise_conv1dbn_0_x_in <= depthwise_conv1d_0_y_out;
    pointwise_conv1dbn_0_enable <= depthwise_conv1d_0_done;

    shortcut_conv1d_clock <= clock;
    inst_${name}_shortcut_conv1d: entity ${work_library_name}.${name}_shortcut_conv1d(rtl)
    port map (
        enable => shortcut_conv1d_enable,
        clock  => shortcut_conv1d_clock,
        x_addr  => shortcut_conv1d_x_addr,
        y_addr  => shortcut_conv1d_y_addr,
        x_in  => shortcut_conv1d_x_in,
        y_out => shortcut_conv1d_y_out,
        done  => shortcut_conv1d_done
    );
    shortcut_conv1d_y_addr <= add_x_1_addr;
    add_x_1 <= shortcut_conv1d_y_out;
    shortcut_conv1d_x_in <= x_in;

    pointwise_conv1dbn_0_clock <= clock;
    inst_${name}_pointwise_conv1dbn_0: entity ${work_library_name}.${name}_pointwise_conv1dbn_0(rtl)
    port map (
        enable => pointwise_conv1dbn_0_enable,
        clock  => pointwise_conv1dbn_0_clock,
        x_addr  => pointwise_conv1dbn_0_x_addr,
        y_addr  => pointwise_conv1dbn_0_y_addr,
        x_in  => pointwise_conv1dbn_0_x_in,
        y_out => pointwise_conv1dbn_0_y_out,
        done  => pointwise_conv1dbn_0_done
    );
    pointwise_conv1dbn_0_y_addr <= depthwise_conv1d_1_x_addr;
    pointwise_conv1dbn_0_relu_x_in <= pointwise_conv1dbn_0_y_out;
    pointwise_conv1dbn_0_relu_enable <= pointwise_conv1dbn_0_done;

    pointwise_conv1dbn_0_relu_clock <= clock;
    inst_${name}_pointwise_conv1dbn_0_relu: entity ${work_library_name}.${name}_pointwise_conv1dbn_0_relu(rtl)
    port map (
        enable => pointwise_conv1dbn_0_relu_enable,
        clock  => pointwise_conv1dbn_0_relu_clock,
        x_in  => pointwise_conv1dbn_0_relu_x_in,
        y_out  => pointwise_conv1dbn_0_relu_y_out
    );

    depthwise_conv1d_1_x_in <= pointwise_conv1dbn_0_relu_y_out;

    depthwise_conv1d_1_clock <= clock;
    depthwise_conv1d_1_enable <= pointwise_conv1dbn_0_done;
    inst_${name}_depthwise_conv1d_1: entity ${work_library_name}.${name}_depthwise_conv1d_1(rtl)
    port map (
        enable => depthwise_conv1d_1_enable,
        clock  => depthwise_conv1d_1_clock,
        x_addr  => depthwise_conv1d_1_x_addr,
        y_addr  => depthwise_conv1d_1_y_addr,
        x_in  => depthwise_conv1d_1_x_in,
        y_out => depthwise_conv1d_1_y_out,
        done  => depthwise_conv1d_1_done
    );

    depthwise_conv1d_1_y_addr <= pointwise_conv1dbn_1_x_addr;
    pointwise_conv1dbn_1_x_in <= depthwise_conv1d_1_y_out;
    pointwise_conv1dbn_1_enable <= depthwise_conv1d_1_done;

    pointwise_conv1dbn_1_clock <= clock;
    inst_${name}_pointwise_conv1dbn_1: entity ${work_library_name}.${name}_pointwise_conv1dbn_1(rtl)
    port map (
        enable => pointwise_conv1dbn_1_enable,
        clock  => pointwise_conv1dbn_1_clock,
        x_addr  => pointwise_conv1dbn_1_x_addr,
        y_addr  => pointwise_conv1dbn_1_y_addr,
        x_in  => pointwise_conv1dbn_1_x_in,
        y_out => pointwise_conv1dbn_1_y_out,
        done  => pointwise_conv1dbn_1_done
    );

    pointwise_conv1dbn_1_y_addr <= add_x_2_addr;
    add_x_2 <= pointwise_conv1dbn_1_y_out;


    add_enable <= pointwise_conv1dbn_1_done and shortcut_conv1d_done;
    add_clock <= clock;
    inst_${name}_add: entity ${work_library_name}.${name}_add(rtl)
    port map (
        enable => add_enable,
        clock  => add_clock,
        x_1_addr  => add_x_1_addr,
        x_1_in  => add_x_1,
        x_2_addr  => add_x_2_addr,
        x_2_in  => add_x_2,
        y_addr  => add_y_addr,
        y_out => add_y_out,
        done  => add_done
    );
    add_y_addr <= y_addr;
    relu_x <= add_y_out;


    relu_enable <= add_done;
    relu_clock <= clock;
    inst_${name}_relu: entity ${work_library_name}.${name}_relu(rtl)
    port map (
        enable => relu_enable,
        clock  => relu_clock,
        x_in  => relu_x,
        y_out  => relu_y
    );
    y_out <= relu_y;
    done <= add_done;

end architecture;
