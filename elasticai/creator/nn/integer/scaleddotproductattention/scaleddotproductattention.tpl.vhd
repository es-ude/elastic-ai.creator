library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        X_1_ADDR_WIDTH : integer := ${x_1_addr_width};
        X_2_ADDR_WIDTH : integer := ${x_2_addr_width};
        Y_SCORE_ADDR_WIDTH : integer := ${y_score_addr_width};
        X_3_ADDR_WIDTH : integer := ${x_3_addr_width};
        Y_ADDR_WIDTH : integer := ${y_addr_width};
        DATA_WIDTH : integer := ${data_width}
    );
    port (
        enable : in std_logic;
        clock : in std_logic;
        x_1_address : out std_logic_vector(X_1_ADDR_WIDTH-1 downto 0);
        x_2_address : out std_logic_vector(X_2_ADDR_WIDTH-1 downto 0);
        x_3_address : out std_logic_vector(X_3_ADDR_WIDTH-1 downto 0);
        y_address : in std_logic_vector(Y_ADDR_WIDTH-1 downto 0);
        x_1 : in std_logic_vector(DATA_WIDTH-1 downto 0);
        x_2 : in std_logic_vector(DATA_WIDTH-1 downto 0);
        x_3 : in std_logic_vector(DATA_WIDTH-1 downto 0);
        y : out std_logic_vector(DATA_WIDTH-1 downto 0);
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
    signal matmul_score_enable : std_logic;
    signal matmul_score_clock : std_logic;
    signal matmul_score_x_1_address : std_logic_vector(X_1_ADDR_WIDTH - 1 downto 0);
    signal matmul_score_x_1 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal matmul_score_x_2_address : std_logic_vector(X_2_ADDR_WIDTH - 1 downto 0);
    signal matmul_score_x_2 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal matmul_score_y_address : std_logic_vector(Y_SCORE_ADDR_WIDTH - 1 downto 0);
    signal matmul_score_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal matmul_score_done : std_logic;
    signal softmax_enable : std_logic;
    signal softmax_clock : std_logic;
    signal softmax_x_address : std_logic_vector(Y_SCORE_ADDR_WIDTH - 1 downto 0);
    signal softmax_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal softmax_y_address : std_logic_vector(Y_SCORE_ADDR_WIDTH - 1 downto 0);
    signal softmax_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal softmax_done : std_logic;
    signal matmul_att_enable : std_logic;
    signal matmul_att_clock : std_logic;
    signal matmul_att_x_1_address : std_logic_vector(Y_SCORE_ADDR_WIDTH - 1 downto 0);
    signal matmul_att_x_1 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal matmul_att_x_2_address : std_logic_vector(X_3_ADDR_WIDTH - 1 downto 0);
    signal matmul_att_x_2 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal matmul_att_y_address : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
    signal matmul_att_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal matmul_att_done : std_logic;
begin
    matmul_score_enable <= enable;
    softmax_enable <= matmul_score_done;
    matmul_att_enable <= softmax_done;
    matmul_score_clock <= clock;
    softmax_clock <= clock;
    matmul_att_clock <= clock;
    x_1_address <= matmul_score_x_1_address;
    x_2_address <= matmul_score_x_2_address;
    x_3_address <= matmul_att_x_2_address;
    matmul_score_y_address <= softmax_x_address;
    softmax_y_address <= matmul_att_x_1_address;
    matmul_att_y_address <= y_address;
    matmul_score_x_1 <= x_1;
    matmul_score_x_2 <= x_2;
    softmax_x <= matmul_score_y;
    matmul_att_x_1 <= softmax_y;
    matmul_att_x_2 <= x_3;
    y <= matmul_att_y;
    done <= matmul_att_done;
    inst_${name}_matmul_score: entity ${work_library_name}.${name}_matmul_score(rtl)
    port map (
        enable => matmul_score_enable,
        clock => matmul_score_clock,
        x_1_address => matmul_score_x_1_address,
        x_1 => matmul_score_x_1,
        x_2_address => matmul_score_x_2_address,
        x_2 => matmul_score_x_2,
        y_address => matmul_score_y_address,
        y => matmul_score_y,
        done => matmul_score_done
    );
    inst_${name}_softmax: entity ${work_library_name}.${name}_softmax(rtl)
    port map (
        enable => softmax_enable,
        clock => softmax_clock,
        x_addr => softmax_x_address,
        x => softmax_x,
        y_address => softmax_y_address,
        y => softmax_y,
        done => softmax_done
    );
    inst_${name}_matmul_att: entity ${work_library_name}.${name}_matmul_att(rtl)
    port map (
        enable => matmul_att_enable,
        clock => matmul_att_clock,
        x_1_address => matmul_att_x_1_address,
        x_1 => matmul_att_x_1,
        x_2_address => matmul_att_x_2_address,
        x_2 => matmul_att_x_2,
        y_address => matmul_att_y_address,
        y => matmul_att_y,
        done => matmul_att_done
    );

end architecture;
