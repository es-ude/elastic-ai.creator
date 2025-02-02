library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        DATA_WIDTH : integer := ${data_width};
        LSTM_CELL_X_1_ADDR_WIDTH : integer := ${lstm_cell_x_1_addr_width};
        LSTM_CELL_X_2_ADDR_WIDTH : integer := ${lstm_cell_x_2_addr_width};
        LSTM_CELL_X_3_ADDR_WIDTH : integer := ${lstm_cell_x_3_addr_width};
        LSTM_CELL_Y_1_ADDR_WIDTH : integer := ${lstm_cell_y_1_addr_width};
        LSTM_CELL_Y_2_ADDR_WIDTH : integer := ${lstm_cell_y_2_addr_width}
    );
    port
    (
        clk : in std_logic;
        reset : in std_logic;
        enable : in std_logic;
        lstm_cell_x_1_addr : in std_logic_vector(LSTM_CELL_X_1_ADDR_WIDTH - 1 downto 0);
        lstm_cell_x_2_addr : in std_logic_vector(LSTM_CELL_X_2_ADDR_WIDTH - 1 downto 0);
        lstm_cell_x_3_addr : in std_logic_vector(LSTM_CELL_X_3_ADDR_WIDTH - 1 downto 0);
        lstm_cell_y_1_addr : in std_logic_vector(LSTM_CELL_Y_1_ADDR_WIDTH - 1 downto 0);
        lstm_cell_y_2_addr : in std_logic_vector(LSTM_CELL_Y_2_ADDR_WIDTH - 1 downto 0);
        lstm_cell_x_1_data : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        lstm_cell_x_2_data : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        lstm_cell_x_3_data : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        lstm_cell_y_1_data : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        lstm_cell_y_2_data : out std_logic_vector(DATA_WIDTH - 1 downto 0)
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
    signal lstm_cell_x_1_address : std_logic_vector(LSTM_CELL_X_1_ADDR_WIDTH - 1 downto 0);
    signal lstm_cell_x_2_address : std_logic_vector(LSTM_CELL_X_2_ADDR_WIDTH - 1 downto 0);
    signal lstm_cell_x_3_address : std_logic_vector(LSTM_CELL_X_3_ADDR_WIDTH - 1 downto 0);
    signal lstm_cell_x_1: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal lstm_cell_x_2: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal lstm_cell_x_3: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal lstm_cell_y_1_address : std_logic_vector(LSTM_CELL_Y_1_ADDR_WIDTH - 1 downto 0);
    signal lstm_cell_y_2_address : std_logic_vector(LSTM_CELL_Y_2_ADDR_WIDTH - 1 downto 0);
    signal lstm_cell_y_1: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal lstm_cell_y_2: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal lstm_cell_done: std_logic;
    begin
    --- TODO: Add your code here

    inst_${name}_lstm_cell: entity ${work_library_name}.${name}_lstm_cell(rtl)
        port map (
            enable => lstm_cell_enable,
            clock  => lstm_cell_clock,
            x_address  => lstm_cell_x_address,
            y_address  => lstm_cell_y_address,
            x  => lstm_cell_x,
            y => lstm_cell_y,
            done  => lstm_cell_done
        );
    end architecture;
