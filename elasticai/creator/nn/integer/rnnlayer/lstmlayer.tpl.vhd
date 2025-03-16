library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        DATA_WIDTH : integer := ${data_width};
        X_1_ADDR_WIDTH : integer := ${x_1_addr_width};
        X_2_ADDR_WIDTH : integer := ${x_2_addr_width};
        X_3_ADDR_WIDTH : integer := ${x_3_addr_width};
        Y_1_ADDR_WIDTH : integer := ${y_1_addr_width};
        Y_2_ADDR_WIDTH : integer := ${y_2_addr_width};
        Y_3_ADDR_WIDTH : integer := ${y_3_addr_width};
        X_1_COUNT : integer := ${x_1_count};
        X_2_COUNT : integer := ${x_2_count};
        X_3_COUNT : integer := ${x_3_count};
        Y_1_COUNT : integer := ${y_1_count};
        Y_2_COUNT : integer := ${y_2_count};
        Y_3_COUNT : integer := ${y_3_count}
    );
    port
    (
        enable : in std_logic;
        clock : in std_logic;
        x_1_addr : in std_logic_vector(X_1_ADDR_WIDTH - 1 downto 0);
        x_2_addr : in std_logic_vector(X_2_ADDR_WIDTH - 1 downto 0);
        x_3_addr : in std_logic_vector(X_3_ADDR_WIDTH - 1 downto 0);
        y_1_addr : in std_logic_vector(Y_1_ADDR_WIDTH - 1 downto 0);
        y_2_addr : in std_logic_vector(Y_2_ADDR_WIDTH - 1 downto 0);
        y_3_addr : in std_logic_vector(Y_3_ADDR_WIDTH - 1 downto 0);
        x_1_data : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        x_2_data : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        x_3_data : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        y_1_data : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        y_2_data : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        done : out std_logic
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
    signal lstm_cell_enable : std_logic;
    signal lstm_cell_clock : std_logic;
    signal x_1_address : std_logic_vector(log2(X_1_COUNT) - 1 downto 0);
    signal x_2_address : std_logic_vector(log2(X_2_COUNT) - 1 downto 0);
    signal x_3_address : std_logic_vector(log2(X_3_COUNT) - 1 downto 0);
    signal x_1: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal x_2: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal x_3: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal y_1_address : std_logic_vector(log2(Y_1_COUNT) - 1 downto 0);
    signal y_2_address : std_logic_vector(log2(Y_2_COUNT) - 1 downto 0);
    signal y_3_address : std_logic_vector(log2(Y_3_COUNT) - 1 downto 0);
    signal y_1: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal y_2: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal y_3: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal lstm_cell_done: std_logic;
    begin
    lstm_cell_enable <= enable;
    done <= lstm_cell_done;
    lstm_cell_clock <= clock;
    x_1_address <= lstm_cell_x_1_address;
    x_2_address <= lstm_cell_x_2_address;
    x_3_address <= lstm_cell_x_3_address;
    y_1_address <= lstm_cell_y_1_address;
    y_2_address <= lstm_cell_y_2_address;
    y_3_address <= lstm_cell_y_3_address;
    lstm_cell_x_1 <= x_1;
    lstm_cell_x_2 <= x_2;
    lstm_cell_x_3 <= x_3;
    y_1 <= lstm_cell_y_1;
    y_2 <= lstm_cell_y_2;
    y_3 <= lstm_cell_y_3;
    inst_${cell_name}: entity ${work_library_name}.${cell_name}(rtl)
        port map (
            enable => lstm_cell_enable,
            clock  => lstm_cell_clock,
            x_1_address  => x_1_address,
            x_2_address  => x_2_address,
            x_3_address  => x_3_address,
            y_1_address  => y_1_address,
            y_2_address  => y_2_address,
            y_3_address  => y_3_address,
            x_1  => x_1,
            x_2  => x_2,
            x_3  => x_3,
            y_1  => y_1,
            y_2  => y_2,
            y_3  => y_3,
            done  => done
        );
    end architecture;
