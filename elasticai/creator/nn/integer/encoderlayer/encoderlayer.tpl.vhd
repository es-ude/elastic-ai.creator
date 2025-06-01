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
        IN_FEATURES : integer := ${in_features};
        OUT_FEATURES : integer := ${out_features}
    );
    port(
        enable: in std_logic;
        clock: in std_logic;
        x_address: out std_logic_vector(X_ADDR_WIDTH-1 downto 0);
        y_address: in std_logic_vector(Y_ADDR_WIDTH-1 downto 0);
        x: in std_logic_vector(DATA_WIDTH-1 downto 0);
        y: out std_logic_vector(DATA_WIDTH-1 downto 0);
        done: out std_logic
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
    signal mha_enable : std_logic;
    signal mha_clock : std_logic;
    signal mha_x_1_address : std_logic_vector(log2(NUM_DIMENSIONS * IN_FEATURES)-1 downto 0);
    signal mha_x_2_address : std_logic_vector(log2(NUM_DIMENSIONS * IN_FEATURES)-1 downto 0);
    signal mha_x_3_address : std_logic_vector(log2(NUM_DIMENSIONS * IN_FEATURES)-1 downto 0);
    signal mha_y_address : std_logic_vector(log2(NUM_DIMENSIONS * OUT_FEATURES)-1 downto 0);
    signal mha_x_1 : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal mha_x_2 : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal mha_x_3 : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal mha_y : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal mha_done : std_logic;
    signal mha_add_enable : std_logic;
    signal mha_add_clock : std_logic;
    signal mha_add_x_1_address : std_logic_vector(log2(NUM_DIMENSIONS * OUT_FEATURES)-1 downto 0);
    signal mha_add_x_2_address : std_logic_vector(log2(NUM_DIMENSIONS * OUT_FEATURES)-1 downto 0);
    signal mha_add_y_address : std_logic_vector(log2(NUM_DIMENSIONS * OUT_FEATURES)-1 downto 0);
    signal mha_add_x_1 : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal mha_add_x_2 : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal mha_add_y : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal mha_add_done : std_logic;
    signal mha_norm_enable : std_logic;
    signal mha_norm_clock : std_logic;
    signal mha_norm_x_address : std_logic_vector(log2(NUM_DIMENSIONS * OUT_FEATURES)-1 downto 0);
    signal mha_norm_y_address : std_logic_vector(log2(NUM_DIMENSIONS * OUT_FEATURES)-1 downto 0);
    signal mha_norm_x : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal mha_norm_y : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal mha_norm_done : std_logic;
    signal ffn_enable : std_logic;
    signal ffn_clock : std_logic;
    signal ffn_x_address : std_logic_vector(log2(NUM_DIMENSIONS * OUT_FEATURES)-1 downto 0);
    signal ffn_y_address : std_logic_vector(log2(NUM_DIMENSIONS * OUT_FEATURES)-1 downto 0);
    signal ffn_x : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal ffn_y : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal ffn_done : std_logic;
    signal ffn_add_enable : std_logic;
    signal ffn_add_clock : std_logic;
    signal ffn_add_x_1_address : std_logic_vector(log2(NUM_DIMENSIONS * OUT_FEATURES)-1 downto 0);
    signal ffn_add_x_2_address : std_logic_vector(log2(NUM_DIMENSIONS * OUT_FEATURES)-1 downto 0);
    signal ffn_add_y_address : std_logic_vector(log2(NUM_DIMENSIONS * OUT_FEATURES)-1 downto 0);
    signal ffn_add_x_1 : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal ffn_add_x_2 : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal ffn_add_y : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal ffn_add_done : std_logic;
    signal ffn_norm_enable : std_logic;
    signal ffn_norm_clock : std_logic;
    signal ffn_norm_x_address : std_logic_vector(log2(NUM_DIMENSIONS * OUT_FEATURES)-1 downto 0);
    signal ffn_norm_y_address : std_logic_vector(log2(NUM_DIMENSIONS * OUT_FEATURES)-1 downto 0);
    signal ffn_norm_x : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal ffn_norm_y : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal ffn_norm_done : std_logic;
begin
    mha_enable <= enable;
    mha_add_enable <= mha_done;
    mha_norm_enable <= mha_add_done;
    ffn_enable <= mha_norm_done;
    ffn_add_enable <= ffn_done;
    ffn_norm_enable <= ffn_add_done;
    done <= ffn_norm_done;
    mha_clock <= clock;
    mha_add_clock <= clock;
    mha_norm_clock <= clock;
    ffn_clock <= clock;
    ffn_add_clock <= clock;
    ffn_norm_clock <= clock;
    x_address <= mha_x_1_address when mha_done = '0' else
            mha_add_x_1_address when mha_done = '1' and mha_add_done = '0' else
            (others => '0');
    mha_y_address <= mha_add_x_2_address;
    mha_add_y_address <= mha_norm_x_address;
    mha_norm_y_address <= ffn_x_address when ffn_done = '0' else
                    ffn_add_x_1_address when ffn_done = '1' and ffn_add_done = '0' else
                    (others => '0');
    ffn_y_address <= ffn_add_x_2_address;
    ffn_add_y_address <= ffn_norm_x_address;
    ffn_norm_y_address <= y_address;
    mha_x_1 <= x;
    mha_x_2 <= x;
    mha_x_3 <= x;
    mha_add_x_1 <= x;
    mha_add_x_2 <= mha_y;
    mha_norm_x  <= mha_add_y;
    ffn_x  <= mha_norm_y;
    ffn_add_x_1 <= mha_norm_y;
    ffn_add_x_2 <= ffn_y;
    ffn_norm_x  <= ffn_add_y;
    y <= ffn_norm_y;
    inst_${name}_mha : entity ${work_library_name}.${mha_name}(rtl)
    port map (
        enable => mha_enable,
        clock => mha_clock,
        x_1_address => mha_x_1_address,
        x_2_address => mha_x_2_address,
        x_3_address => mha_x_3_address,
        y_address => mha_y_address,
        x_1 => mha_x_1,
        x_2 => mha_x_2,
        x_3 => mha_x_3,
        y => mha_y,
        done => mha_done
    );
    inst_${name}_mha_add : entity ${work_library_name}.${mha_add_name}(rtl)
    port map (
        enable => mha_add_enable,
        clock => mha_add_clock,
        x_1_address => mha_add_x_1_address,
        x_2_address => mha_add_x_2_address,
        y_address => mha_add_y_address,
        x_1 => mha_add_x_1,
        x_2 => mha_add_x_2,
        y => mha_add_y,
        done => mha_add_done
    );
    inst_${name}_mha_norm : entity ${work_library_name}.${mha_norm_name}(rtl)
    port map (
        enable => mha_norm_enable,
        clock => mha_norm_clock,
        x_address => mha_norm_x_address,
        y_address => mha_norm_y_address,
        x => mha_norm_x,
        y => mha_norm_y,
        done => mha_norm_done
    );
    inst_${name}_ffn : entity ${work_library_name}.${ffn_name}(rtl)
    port map (
        enable => ffn_enable,
        clock => ffn_clock,
        x_address => ffn_x_address,
        y_address => ffn_y_address,
        x => ffn_x,
        y => ffn_y,
        done => ffn_done
    );
    inst_${name}_ffn_add : entity ${work_library_name}.${ffn_add_name}(rtl)
    port map (
        enable => ffn_add_enable,
        clock => ffn_add_clock,
        x_1_address => ffn_add_x_1_address,
        x_2_address => ffn_add_x_2_address,
        y_address => ffn_add_y_address,
        x_1 => ffn_add_x_1,
        x_2 => ffn_add_x_2,
        y => ffn_add_y,
        done => ffn_add_done
    );
    inst_${name}_ffn_norm : entity ${work_library_name}.${ffn_norm_name}(rtl)
    port map (
        enable => ffn_norm_enable,
        clock => ffn_norm_clock,
        x_address => ffn_norm_x_address,
        x => ffn_norm_x ,
        y_address => ffn_norm_y_address,
        y => ffn_norm_y,
        done => ffn_norm_done
    );
end architecture;
