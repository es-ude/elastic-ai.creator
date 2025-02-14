library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        DATA_WIDTH : integer := ${data_width};
        Q_LINEAR_X_ADDR_WIDTH : integer := ${q_linear_x_addr_width};
        Q_LINEAR_Y_ADDR_WIDTH : integer := ${q_linear_y_addr_width};
        K_LINEAR_X_ADDR_WIDTH : integer := ${k_linear_x_addr_width};
        K_LINEAR_Y_ADDR_WIDTH : integer := ${k_linear_y_addr_width};
        V_LINEAR_X_ADDR_WIDTH : integer := ${v_linear_x_addr_width};
        V_LINEAR_Y_ADDR_WIDTH : integer := ${v_linear_y_addr_width};
        INNER_ATTN_Y_ADDR_WIDTH : integer := ${inner_attn_y_address_width};
        OUTPUT_LINEAR_Y_ADDR_WIDTH : integer := ${output_linear_y_addr_width}
    );
    port (
        enable : in std_logic;
        clock : in std_logic;
        x_1_address : out std_logic_vector(Q_LINEAR_X_ADDR_WIDTH-1 downto 0);
        x_2_address : out std_logic_vector(K_LINEAR_X_ADDR_WIDTH-1 downto 0);
        x_3_address : out std_logic_vector(V_LINEAR_X_ADDR_WIDTH-1 downto 0);
        y_address : in std_logic_vector(OUTPUT_LINEAR_Y_ADDR_WIDTH-1 downto 0);
        x_1 : in std_logic_vector(DATA_WIDTH-1 downto 0);
        x_2 : in std_logic_vector(DATA_WIDTH-1 downto 0);
        x_3 : in std_logic_vector(DATA_WIDTH-1 downto 0);
        y : out std_logic_vector(DATA_WIDTH-1 downto 0);
        done : out std_logic
    );
end ${name};
architecture rtl of ${name} is
    signal q_linear_enable : std_logic;
    signal q_linear_clock : std_logic;
    signal q_linear_x_address : std_logic_vector(Q_LINEAR_X_ADDR_WIDTH-1 downto 0);
    signal q_linear_y_address : std_logic_vector(Q_LINEAR_Y_ADDR_WIDTH-1 downto 0);
    signal q_linear_x : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal q_linear_y : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal q_linear_done : std_logic;
    signal k_linear_enable : std_logic;
    signal k_linear_clock : std_logic;
    signal k_linear_x_address : std_logic_vector(K_LINEAR_X_ADDR_WIDTH-1 downto 0);
    signal k_linear_y_address : std_logic_vector(K_LINEAR_Y_ADDR_WIDTH-1 downto 0);
    signal k_linear_x : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal k_linear_y : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal k_linear_done : std_logic;
    signal v_linear_enable : std_logic;
    signal v_linear_clock : std_logic;
    signal v_linear_x_address : std_logic_vector(V_LINEAR_X_ADDR_WIDTH-1 downto 0);
    signal v_linear_y_address : std_logic_vector(V_LINEAR_Y_ADDR_WIDTH-1 downto 0);
    signal v_linear_x : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal v_linear_y : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal v_linear_done : std_logic;
    signal inner_attn_enable : std_logic;
    signal inner_attn_clock : std_logic;
    signal inner_attn_x_1_address : std_logic_vector(Q_LINEAR_Y_ADDR_WIDTH-1 downto 0);
    signal inner_attn_x_2_address : std_logic_vector(K_LINEAR_Y_ADDR_WIDTH-1 downto 0);
    signal inner_attn_x_3_address : std_logic_vector(V_LINEAR_Y_ADDR_WIDTH-1 downto 0);
    signal inner_attn_y_address : std_logic_vector(INNER_ATTN_Y_ADDR_WIDTH-1 downto 0);
    signal inner_attn_x_1 : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal inner_attn_x_2 : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal inner_attn_x_3 : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal inner_attn_y : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal inner_attn_done : std_logic;
    signal output_linear_enable : std_logic;
    signal output_linear_clock : std_logic;
    signal output_linear_x_address : std_logic_vector(INNER_ATTN_Y_ADDR_WIDTH-1 downto 0);
    signal output_linear_y_address : std_logic_vector(OUTPUT_LINEAR_Y_ADDR_WIDTH-1 downto 0);
    signal output_linear_x : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal output_linear_y : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal output_linear_done : std_logic;
begin
    q_linear_enable <= enable;
    k_linear_enable <= enable;
    v_linear_enable <= enable;
    inner_attn_enable <= q_linear_done;
    output_linear_enable <= inner_attn_done;
    q_linear_clock <= clock;
    k_linear_clock <= clock;
    v_linear_clock <= clock;
    inner_attn_clock <= clock;
    output_linear_clock <= clock;
    x_1_address <= q_linear_x_address;
    x_2_address <= k_linear_x_address;
    x_3_address <= v_linear_x_address;
    q_linear_y_address <= inner_attn_x_1_address;
    k_linear_y_address <= inner_attn_x_2_address;
    v_linear_y_address <= inner_attn_x_3_address;
    inner_attn_y_address <= output_linear_x_addr;
    output_linear_y_addr <= y_address;
    q_linear_x <= x_1;
    k_linear_x <= x_2;
    v_linear_x <= x_3;
    inner_attn_x_1 <= q_linear_y;
    inner_attn_x_2 <= k_linear_y;
    inner_attn_x_3 <= v_linear_y;
    output_linear_x <= inner_attn_y;
    y <= output_linear_y;
    done <= output_linear_done;
    inst_${name}_q_linear : entity ${work_library_name}.${name}_q_linear(rtl)
    port map (
        enable => q_linear_enable,
        clock => q_linear_clock,
        x_addr => q_linear_x_address,
        y_address => q_linear_y_address,
        x => q_linear_x,
        y => q_linear_y,
        done => q_linear_done
    );
    inst_${name}_k_linear : entity ${work_library_name}.${name}_k_linear(rtl)
    port map (
        enable => k_linear_enable,
        clock => k_linear_clock,
        x_addr => k_linear_x_address,
        y_address => k_linear_y_address,
        x => k_linear_x,
        y => k_linear_y,
        done => k_linear_done
    );
    inst_${name}_v_linear : entity ${work_library_name}.${name}_v_linear(rtl)
    port map (
        enable => v_linear_enable,
        clock => v_linear_clock,
        x_addr => v_linear_x_address,
        y_address => v_linear_y_address,
        x => v_linear_x,
        y => v_linear_y,
        done => v_linear_done
    );
    inst_${name}_inner_attn : entity ${work_library_name}.${name}_inner_attn(rtl)
    port map (
        enable => inner_attn_enable,
        clock => inner_attn_clock,
        x_1_address => inner_attn_x_1_address,
        x_2_address => inner_attn_x_2_address,
        x_3_address => inner_attn_x_3_address,
        y_address => inner_attn_y_address,
        x_1 => inner_attn_x_1,
        x_2 => inner_attn_x_2,
        x_3 => inner_attn_x_3,
        y => inner_attn_y,
        done => inner_attn_done
    );
    inst_${name}_output_linear : entity ${work_library_name}.${name}_output_linear(rtl)
    port map (
        enable => output_linear_enable,
        clock => output_linear_clock,
        x_addr => output_linear_x_addr,
        y_address => output_linear_y_addr,
        x => output_linear_x,
        y => output_linear_y,
        done => output_linear_done
    );
end architecture;
