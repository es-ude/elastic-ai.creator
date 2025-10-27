library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        X_DATA_WIDTH : integer := ${x_data_width};
        FC1_Y_DATA_WIDTH : integer := ${fc1_y_data_width};
        RELU_X_DATA_WIDTH : integer := ${relu_x_data_width};
        RELU_Y_DATA_WIDTH : integer := ${relu_y_data_width};
        FC2_X_DATA_WIDTH : integer := ${fc2_x_data_width};
        Y_DATA_WIDTH : integer := ${y_data_width};
        X_ADDR_WIDTH : integer := ${x_addr_width};
        FC1_Y_ADDR_WIDTH : integer := ${fc1_y_addr_width};
        Y_ADDR_WIDTH : integer := ${y_addr_width}
    ) ;
    port (
        enable : in std_logic;
        clock : in std_logic;
        x_address : out std_logic_vector(X_ADDR_WIDTH - 1 downto 0);
        x : in std_logic_vector(X_DATA_WIDTH - 1 downto 0);
        y_address : in std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
        y : out std_logic_vector(Y_DATA_WIDTH - 1 downto 0);
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

    signal fc1_enable : std_logic;
    signal fc1_clock : std_logic;
    signal fc1_x_address : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal fc1_x: std_logic_vector(X_DATA_WIDTH - 1 downto 0);
    signal fc1_y_address : std_logic_vector(FC1_Y_ADDR_WIDTH-1 downto 0);
    signal fc1_y: std_logic_vector(FC1_Y_DATA_WIDTH - 1 downto 0);
    signal fc1_done : std_logic;

    signal relu_enable : std_logic;
    signal relu_clock : std_logic;
    signal relu_x: std_logic_vector(RELU_X_DATA_WIDTH - 1 downto 0);
    signal relu_y : std_logic_vector(RELU_Y_DATA_WIDTH - 1 downto 0);

    signal fc2_enable : std_logic;
    signal fc2_clock : std_logic;
    signal fc2_x_address : std_logic_vector(FC1_Y_ADDR_WIDTH-1 downto 0);
    signal fc2_x: std_logic_vector(FC2_X_DATA_WIDTH - 1 downto 0);
    signal fc2_y_address : std_logic_vector(Y_ADDR_WIDTH-1 downto 0);
    signal fc2_y: std_logic_vector(Y_DATA_WIDTH - 1 downto 0);
    signal fc2_done : std_logic;
    begin
    fc1_enable <= enable;
    relu_enable <= fc1_done;
    fc2_enable <= fc1_done;
    fc1_clock <= clock;
    relu_clock <= clock;
    fc2_clock <= clock;
    x_address <= fc1_x_address;
    fc1_y_address <= fc2_x_address;
    fc2_y_address <= y_address;
    fc1_x <= x;
    relu_x <= fc1_y;
    fc2_x <= relu_y;
    y <= fc2_y;
    done <= fc2_done;
    inst_${name}_fc1: entity ${work_library_name}.${fc1_name}(rtl)
    port map (
        enable => fc1_enable,
        clock  => fc1_clock,
        x_address  => fc1_x_address,
        y_address  => fc1_y_address,
        x  => fc1_x,
        y => fc1_y,
        done  => fc1_done
    );
    inst_${name}_relu: entity ${work_library_name}.${relu_name}(rtl)
    port map (
        enable => relu_enable,
        clock  => relu_clock,
        x  => relu_x,
        y => relu_y
    );
    inst_${name}_fc2: entity ${work_library_name}.${fc2_name}(rtl)
    port map (
        enable => fc2_enable,
        clock  => fc2_clock,
        x_address  => fc2_x_address,
        y_address  => fc2_y_address,
        x  => fc2_x,
        y => fc2_y,
        done  => fc2_done
    );
end architecture;
