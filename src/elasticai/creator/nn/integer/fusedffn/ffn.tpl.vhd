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
        FC1_IN_FEATURES : integer := ${fc1_in_features};
        FC1_OUT_FEATURES : integer := ${fc1_out_features};
        FC2_OUT_FEATURES : integer := ${fc2_out_features}
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
    signal fc1_enable : std_logic;
    signal fc1_clock : std_logic;
    signal fc1_x_address : std_logic_vector(log2(FC1_IN_FEATURES * NUM_DIMENSIONS)-1 downto 0);
    signal fc1_x: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal fc1_y_address : std_logic_vector(log2(FC1_OUT_FEATURES)-1 downto 0);
    signal fc1_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal fc1_done : std_logic;
    signal fc1_valid : std_logic;
    signal fc1_ready : std_logic;
    signal fc2_enable : std_logic;
    signal fc2_clock : std_logic;
    signal fc2_x_address : std_logic_vector(log2(FC1_OUT_FEATURES)-1 downto 0);
    signal fc2_x: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal fc2_y_address : std_logic_vector(log2(FC2_OUT_FEATURES * NUM_DIMENSIONS)-1 downto 0);
    signal fc2_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal fc2_done : std_logic;
    signal fc2_valid : std_logic;
    signal fc2_ready : std_logic;
    begin
    fc1_enable <= enable;
    fc2_enable <= enable;
    fc1_clock <= clock;
    fc2_clock <= clock;
    x_address <= fc1_x_address;
    fc1_y_address <= fc2_x_address;
    fc2_y_address <= y_address;
    fc1_x <= x;
    fc2_x <= fc1_y;
    y <= fc2_y;
    done <= fc2_done;
    fc2_valid <= fc1_valid;
    fc1_ready <= fc2_ready;
    inst_${name}_fc1: entity ${work_library_name}.${fc1relu_name}(rtl)
    port map (
        enable => fc1_enable,
        clock  => fc1_clock,
        x_address  => fc1_x_address,
        y_address  => fc1_y_address,
        x  => fc1_x,
        y => fc1_y,
        done  => fc1_done,
        valid => fc1_valid, -- out
        ready => fc1_ready  -- in
    );
    inst_${name}_fc2: entity ${work_library_name}.${fc2_name}(rtl)
    port map (
        enable => fc2_enable,
        clock  => fc2_clock,
        x_address  => fc2_x_address,
        y_address  => fc2_y_address,
        x  => fc2_x,
        y => fc2_y,
        done  => fc2_done,
        valid => fc2_valid, -- in
        ready => fc2_ready  -- out
    );
end architecture;
