library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        DATA_WIDTH : integer := ${data_width};
        CONCATENATE_X_1_ADDR_WIDTH : integer := ${concatenate_x_1_addr_width};
        CONCATENATE_X_2_ADDR_WIDTH : integer := ${concatenate_x_2_addr_width};
        CONCATENATE_Y_ADDR_WIDTH : integer := ${concatenate_y_addr_width};
        F_GATE_LINEAR_X_ADDR_WIDTH : integer := ${f_gate_linear_x_addr_width};
        F_GATE_LINEAR_Y_ADDR_WIDTH : integer := ${f_gate_linear_y_addr_width};
        C_GATE_LINEAR_X_ADDR_WIDTH : integer := ${i_gate_linear_x_addr_width};
        C_GATE_LINEAR_Y_ADDR_WIDTH : integer := ${c_gate_linear_y_addr_width};
        I_GATE_LINEAR_X_ADDR_WIDTH : integer := ${i_gate_linear_x_addr_width};
        I_GATE_LINEAR_Y_ADDR_WIDTH : integer := ${i_gate_linear_y_addr_width};
        O_GATE_LINEAR_X_ADDR_WIDTH : integer := ${o_gate_linear_x_addr_width};
        O_GATE_LINEAR_Y_ADDR_WIDTH : integer := ${o_gate_linear_y_addr_width};
        I_SIGMOID_X_ADDR_WIDTH : integer := ${i_sigmoid_x_addr_width};
        I_SIGMOID_Y_ADDR_WIDTH : integer := ${i_sigmoid_y_addr_width};
        F_SIGMOID_X_ADDR_WIDTH : integer := ${f_sigmoid_x_addr_width};
        F_SIGMOID_Y_ADDR_WIDTH : integer := ${f_sigmoid_y_addr_width};
        O_SIGMOID_X_ADDR_WIDTH : integer := ${o_sigmoid_x_addr_width};
        O_SIGMOID_Y_ADDR_WIDTH : integer := ${o_sigmoid_y_addr_width};
        C_TANH_X_ADDR_WIDTH : integer := ${c_tanh_x_addr_width};
        C_TANH_Y_ADDR_WIDTH : integer := ${c_tanh_y_addr_width};
        C_NEXT_TANH_X_ADDR_WIDTH : integer := ${c_next_tanh_x_addr_width};
        C_NEXT_TANH_Y_ADDR_WIDTH : integer := ${c_next_tanh_y_addr_width};
        C_NEXT_ADDITION_X_ADDR_WIDTH : integer := ${c_next_addition_x_addr_width};
        C_NEXT_ADDITION_Y_ADDR_WIDTH : integer := ${c_next_addition_y_addr_width};
        FC_HAMADARD_PRODUCT_X_1_ADDR_WIDTH : integer := ${fc_hamadard_product_x_1_addr_width};
        FC_HAMADARD_PRODUCT_X_2_ADDR_WIDTH : integer := ${fc_hamadard_product_x_2_addr_width};
        FC_HAMADARD_PRODUCT_Y_ADDR_WIDTH : integer := ${fc_hamadard_product_y_addr_width};
        IC_HAMADARD_PRODUCT_X_1_ADDR_WIDTH : integer := ${ic_hamadard_product_x_1_addr_width};
        IC_HAMADARD_PRODUCT_X_2_ADDR_WIDTH : integer := ${ic_hamadard_product_x_2_addr_width};
        IC_HAMADARD_PRODUCT_Y_ADDR_WIDTH : integer := ${ic_hamadard_product_y_addr_width};
        OC_HAMADARD_PRODUCT_X_1_ADDR_WIDTH : integer := ${oc_hamadard_product_x_1_addr_width};
        OC_HAMADARD_PRODUCT_X_2_ADDR_WIDTH : integer := ${oc_hamadard_product_x_2_addr_width};
        OC_HAMADARD_PRODUCT_Y_ADDR_WIDTH : integer := ${oc_hamadard_product_y_addr_width}
    ) ;
    port (
        enable : in std_logic;
        clock : in std_logic;
        x_1_address : out std_logic_vector(CONCATENATE_X_1_ADDR_WIDTH - 1 downto 0);         -- last time step
        x_2_address : out std_logic_vector(CONCATENATE_X_2_ADDR_WIDTH - 1 downto 0);         -- last hidden state
        x_3_address : out std_logic_vector(FC_HAMADARD_PRODUCT_X_2_ADDR_WIDTH - 1 downto 0); -- last cell state
        x_1 : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        x_2 : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        x_3 : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        y_1_address : in std_logic_vector(OC_HAMADARD_PRODUCT_Y_ADDR_WIDTH - 1 downto 0); -- next hidden state
        y_2_address : in std_logic_vector(C_NEXT_ADDITION_Y_ADDR_WIDTH - 1 downto 0);     -- next cell state
        y_1 : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        y_2 : out std_logic_vector(DATA_WIDTH - 1 downto 0);
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
    signal concatenate_enable : std_logic;
    signal concatenate_clock : std_logic;
    signal concatenate_x_1_address : std_logic_vector(CONCATENATE_X_1_ADDR_WIDTH - 1 downto 0);
    signal concatenate_x_2_address : std_logic_vector(CONCATENATE_X_2_ADDR_WIDTH - 1 downto 0);
    signal concatenate_x_1 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal concatenate_x_2 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal concatenate_y_address : std_logic_vector(CONCATENATE_Y_ADDR_WIDTH - 1 downto 0);
    signal concatenate_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal concatenate_done : std_logic;
    signal f_gate_linear_enable : std_logic;
    signal f_gate_linear_clock : std_logic;
    signal f_gate_linear_x_address : std_logic_vector(F_GATE_LINEAR_X_ADDR_WIDTH - 1 downto 0);
    signal f_gate_linear_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal f_gate_linear_y_address : std_logic_vector(F_GATE_LINEAR_Y_ADDR_WIDTH - 1 downto 0);
    signal f_gate_linear_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal f_gate_linear_done : std_logic;
    signal c_gate_linear_enable : std_logic;
    signal c_gate_linear_clock : std_logic;
    signal c_gate_linear_x_address : std_logic_vector(C_GATE_LINEAR_X_ADDR_WIDTH - 1 downto 0);
    signal c_gate_linear_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal c_gate_linear_y_address : std_logic_vector(C_GATE_LINEAR_Y_ADDR_WIDTH - 1 downto 0);
    signal c_gate_linear_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal c_gate_linear_done : std_logic;
    signal i_gate_linear_enable : std_logic;
    signal i_gate_linear_clock : std_logic;
    signal i_gate_linear_x_address : std_logic_vector(I_GATE_LINEAR_X_ADDR_WIDTH - 1 downto 0);
    signal i_gate_linear_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal i_gate_linear_y_address : std_logic_vector(I_GATE_LINEAR_Y_ADDR_WIDTH - 1 downto 0);
    signal i_gate_linear_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal i_gate_linear_done : std_logic;
    signal o_gate_linear_enable : std_logic;
    signal o_gate_linear_clock : std_logic;
    signal o_gate_linear_x_address : std_logic_vector(O_GATE_LINEAR_X_ADDR_WIDTH - 1 downto 0);
    signal o_gate_linear_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal o_gate_linear_y_address : std_logic_vector(O_GATE_LINEAR_Y_ADDR_WIDTH - 1 downto 0);
    signal o_gate_linear_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal o_gate_linear_done : std_logic;
    signal i_sigmoid_enable : std_logic;
    signal i_sigmoid_clock : std_logic;
    signal i_sigmoid_x_address : std_logic_vector(I_SIGMOID_X_ADDR_WIDTH - 1 downto 0);
    signal i_sigmoid_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal i_sigmoid_y_address : std_logic_vector(I_SIGMOID_Y_ADDR_WIDTH - 1 downto 0);
    signal i_sigmoid_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal i_sigmoid_done : std_logic;
    signal f_sigmoid_enable : std_logic;
    signal f_sigmoid_clock : std_logic;
    signal f_sigmoid_x_address : std_logic_vector(F_SIGMOID_X_ADDR_WIDTH - 1 downto 0);
    signal f_sigmoid_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal f_sigmoid_y_address : std_logic_vector(F_SIGMOID_Y_ADDR_WIDTH - 1 downto 0);
    signal f_sigmoid_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal f_sigmoid_done : std_logic;
    signal o_sigmoid_enable : std_logic;
    signal o_sigmoid_clock : std_logic;
    signal o_sigmoid_x_address : std_logic_vector(O_SIGMOID_X_ADDR_WIDTH - 1 downto 0);
    signal o_sigmoid_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal o_sigmoid_y_address : std_logic_vector(O_SIGMOID_Y_ADDR_WIDTH - 1 downto 0);
    signal o_sigmoid_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal o_sigmoid_done : std_logic;
    signal c_tanh_enable : std_logic;
    signal c_tanh_clock : std_logic;
    signal c_tanh_x_address : std_logic_vector(C_TANH_X_ADDR_WIDTH - 1 downto 0);
    signal c_tanh_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal c_tanh_y_address : std_logic_vector(C_TANH_Y_ADDR_WIDTH - 1 downto 0);
    signal c_tanh_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal c_tanh_done : std_logic;
    signal c_next_tanh_enable : std_logic;
    signal c_next_tanh_clock : std_logic;
    signal c_next_tanh_x_address : std_logic_vector(C_NEXT_TANH_X_ADDR_WIDTH - 1 downto 0);
    signal c_next_tanh_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal c_next_tanh_y_address : std_logic_vector(C_NEXT_TANH_Y_ADDR_WIDTH - 1 downto 0);
    signal c_next_tanh_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal c_next_tanh_done : std_logic;
    signal c_next_addition_enable : std_logic;
    signal c_next_addition_clock : std_logic;
    signal c_next_addition_x_1_address : std_logic_vector(C_NEXT_ADDITION_X_1_ADDR_WIDTH - 1 downto 0);
    signal c_next_addition_x_2_address : std_logic_vector(C_NEXT_ADDITION_X_2_ADDR_WIDTH - 1 downto 0);
    signal c_next_addition_x_1 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal c_next_addition_x_2 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal c_next_addition_y_address : std_logic_vector(C_NEXT_ADDITION_Y_ADDR_WIDTH - 1 downto 0);
    signal c_next_addition_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal c_next_addition_done : std_logic;
    signal fc_hamadard_product_enable : std_logic;
    signal fc_hamadard_product_clock : std_logic;
    signal fc_hamadard_product_x_1_address : std_logic_vector(FC_HAMADARD_PRODUCT_X_1_ADDR_WIDTH - 1 downto 0);
    signal fc_hamadard_product_x_2_address : std_logic_vector(FC_HAMADARD_PRODUCT_X_2_ADDR_WIDTH - 1 downto 0);
    signal fc_hamadard_product_x_1 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal fc_hamadard_product_x_2 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal fc_hamadard_product_y_address : std_logic_vector(FC_HAMADARD_PRODUCT_Y_ADDR_WIDTH - 1 downto 0);
    signal fc_hamadard_product_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal fc_hamadard_product_done : std_logic;
    signal ic_hamadard_product_enable : std_logic;
    signal ic_hamadard_product_clock : std_logic;
    signal ic_hamadard_product_x_1_address : std_logic_vector(IC_HAMADARD_PRODUCT_X_ADDR_WIDTH - 1 downto 0);
    signal ic_hamadard_product_x_2_address : std_logic_vector(IC_HAMADARD_PRODUCT_X_ADDR_WIDTH - 1 downto 0);
    signal ic_hamadard_product_x_1 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal ic_hamadard_product_x_2 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal ic_hamadard_product_y_address : std_logic_vector(IC_HAMADARD_PRODUCT_Y_ADDR_WIDTH - 1 downto 0);
    signal ic_hamadard_product_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal ic_hamadard_product_done : std_logic;
    signal oc_hamadard_product_enable : std_logic;
    signal oc_hamadard_product_clock : std_logic;
    signal oc_hamadard_product_x_1_address : std_logic_vector(OC_HAMADARD_PRODUCT_X_ADDR_WIDTH - 1 downto 0);
    signal oc_hamadard_product_x_2_address : std_logic_vector(OC_HAMADARD_PRODUCT_X_ADDR_WIDTH - 1 downto 0);
    signal oc_hamadard_product_x_1 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal oc_hamadard_product_x_2 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal oc_hamadard_product_y_address : std_logic_vector(OC_HAMADARD_PRODUCT_Y_ADDR_WIDTH - 1 downto 0);
    signal oc_hamadard_product_y: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal oc_hamadard_product_done : std_logic;
    begin
    concetenate_enable <= enable;
    f_gate_linear_enable <= concetenate_done;
    c_gate_linear_enable <= concetenate_done;
    i_gate_linear_enable <= concetenate_done;
    o_gate_linear_enable <= concetenate_done;
    i_sigmoid_enable <= i_gate_linear_done;
    f_sigmoid_enable <= f_gate_linear_done;
    o_sigmoid_enable <= o_gate_linear_done;
    c_tanh_enable <= c_gate_linear_done;
    fc_hamadard_product_enable <= f_sigmoid_done and c_tanh_done;
    ic_hamadard_product_enable <= i_sigmoid_done and c_tanh_done;
    c_next_addition_enable <= fc_hamadard_product_done and ic_hamadard_product_done;
    c_next_tanh_enable <= c_next_addition_done;
    oc_hamadard_product_enable <= o_sigmoid_done and c_next_tanh_done;
    done <= oc_hamadard_product_done;

    concetenate_clock <= clock;
    f_gate_linear_clock <= clock;
    c_gate_linear_clock <= clock;
    i_gate_linear_clock <= clock;
    o_gate_linear_clock <= clock;
    i_sigmoid_clock <= clock;
    f_sigmoid_clock <= clock;
    o_sigmoid_clock <= clock;
    c_tanh_clock <= clock;
    c_next_tanh_clock <= clock;
    c_next_addition_clock <= clock;
    fc_hamadard_product_clock <= clock;
    ic_hamadard_product_clock <= clock;
    oc_hamadard_product_clock <= clock;

    x_1_address <= concetenate_x_1_address;
    x_2_address <= concetenate_x_2_address;
    concetenate_y_address <= f_gate_linear_x_address;
    concetenate_y_address <= c_gate_linear_x_address;
    concetenate_y_address <= i_gate_linear_x_address;
    concetenate_y_address <= o_gate_linear_x_address;
    f_gate_linear_y_address <= f_sigmoid_x_address;
    c_gate_linear_y_address <= c_tanh_x_address;
    i_gate_linear_y_address <= i_sigmoid_x_address;
    o_gate_linear_y_address <= o_sigmoid_x_address;
    f_sigmoid_y_address <= fc_hamadard_product_x_1_address;
    x_3_address <= fc_hamadard_product_x_2_address;
    i_sigmoid_y_address <= ic_hamadard_product_x_1_address;
    c_tanh_y_address <= ic_hamadard_product_x_2_address;
    c_next_addition_x_1_address <= fc_hamadard_product_y_address;
    c_next_addition_x_2_address <= ic_hamadard_product_y_address;
    c_next_tanh_x_address <= c_next_addition_y_address;
    oc_hamadard_product_x_1_address <= o_sigmoid_y_address;
    oc_hamadard_product_x_2_address <= c_next_tanh_y_address;
    y_1_address <= oc_hamadard_product_y_address;
    y_2_address <= c_next_addition_y_address;
    concetenate_x_1 <= x_1;
    concetenate_x_2 <= x_2;
    f_gate_linear_x <= concetenate_y;
    c_gate_linear_x <= concetenate_y;
    i_gate_linear_x <= concetenate_y;
    o_gate_linear_x <= concetenate_y;
    i_sigmoid_x <= i_gate_linear_y;
    f_sigmoid_x <= f_gate_linear_y;
    o_sigmoid_x <= o_gate_linear_y;
    c_tanh_x <= c_gate_linear_y;
    fc_hamadard_product_x_1 <= f_sigmoid_y;
    fc_hamadard_product_x_2 <= x_3;
    ic_hamadard_product_x_1 <= i_sigmoid_y;
    ic_hamadard_product_x_2 <= c_tanh_y;
    c_next_addition_x_1 <= fc_hamadard_product_y;
    c_next_addition_x_2 <= ic_hamadard_product_y;
    c_next_tanh_x <= c_next_addition_y;
    oc_hamadard_product_x_1 <= o_sigmoid_y;
    oc_hamadard_product_x_2 <= c_next_tanh_y;
    y_1 <= oc_hamadard_product_y;
    y_2 <= c_next_addition_y;
    inst_${name}_concetenate: entity ${work_library_name}.${name}_concatenate(rtl)
    port map (
        enable => concetenate_enable,
        clock  => concetenate_clock,
        x_1_address  => concetenate_x_1_address,
        x_2_address  => concetenate_x_2_address,
        y_address  => concetenate_y_address,
        x_1  => concetenate_x_1,
        x_2  => concetenate_x_2,
        y => concetenate_y,
        done  => concetenate_done
    );
    inst_${name}_f_gate_linear: entity ${work_library_name}.${name}_f_gate_linear(rtl)
    port map (
        enable => f_gate_linear_enable,
        clock  => f_gate_linear_clock,
        x_address  => f_gate_linear_x_address,
        y_address  => f_gate_linear_y_address,
        x  => f_gate_linear_x,
        y => f_gate_linear_y,
        done  => f_gate_linear_done
    );
    inst_${name}_c_gate_linear: entity ${work_library_name}.${name}_c_gate_linear(rtl)
    port map (
        enable => c_gate_linear_enable,
        clock  => c_gate_linear_clock,
        x_address  => c_gate_linear_x_address,
        y_address  => c_gate_linear_y_address,
        x  => c_gate_linear_x,
        y => c_gate_linear_y,
        done  => c_gate_linear_done
    );
    inst_${name}_i_gate_linear: entity ${work_library_name}.${name}_i_gate_linear(rtl)
    port map (
        enable => i_gate_linear_enable,
        clock  => i_gate_linear_clock,
        x_address  => i_gate_linear_x_address,
        y_address  => i_gate_linear_y_address,
        x  => i_gate_linear_x,
        y => i_gate_linear_y,
        done  => i_gate_linear_done
    );
    inst_${name}_o_gate_linear: entity ${work_library_name}.${name}_o_gate_linear(rtl)
    port map (
        enable => o_gate_linear_enable,
        clock  => o_gate_linear_clock,
        x_address  => o_gate_linear_x_address,
        y_address  => o_gate_linear_y_address,
        x  => o_gate_linear_x,
        y => o_gate_linear_y,
        done  => o_gate_linear_done
    );
    inst_${name}_i_sigmoid: entity ${work_library_name}.${name}_i_sigmoid(rtl)
    port map (
        enable => i_sigmoid_enable,
        clock  => i_sigmoid_clock,
        x_address  => i_sigmoid_x_address,
        y_address  => i_sigmoid_y_address,
        x  => i_sigmoid_x,
        y => i_sigmoid_y,
        done  => i_sigmoid_done
    );
    inst_${name}_f_sigmoid: entity ${work_library_name}.${name}_f_sigmoid(rtl)
    port map (
        enable => f_sigmoid_enable,
        clock  => f_sigmoid_clock,
        x_address  => f_sigmoid_x_address,
        y_address  => f_sigmoid_y_address,
        x  => f_sigmoid_x,
        y => f_sigmoid_y,
        done  => f_sigmoid_done
    );
    inst_${name}_o_sigmoid: entity ${work_library_name}.${name}_o_sigmoid(rtl)
    port map (
        enable => o_sigmoid_enable,
        clock  => o_sigmoid_clock,
        x_address  => o_sigmoid_x_address,
        y_address  => o_sigmoid_y_address,
        x  => o_sigmoid_x,
        y => o_sigmoid_y,
        done  => o_sigmoid_done
    );
    inst_${name}_c_tanh: entity ${work_library_name}.${name}_c_tanh(rtl)
    port map (
        enable => c_tanh_enable,
        clock  => c_tanh_clock,
        x_address  => c_tanh_x_address,
        y_address  => c_tanh_y_address,
        x  => c_tanh_x,
        y => c_tanh_y,
        done  => c_tanh_done
    );
    inst_${name}_c_next_tanh: entity ${work_library_name}.${name}_c_next_tanh(rtl)
    port map (
        enable => c_next_tanh_enable,
        clock  => c_next_tanh_clock,
        x_address  => c_next_tanh_x_address,
        y_address  => c_next_tanh_y_address,
        x  => c_next_tanh_x,
        y => c_next_tanh_y,
        done  => c_next_tanh_done
    );
    inst_${name}_c_next_addition: entity ${work_library_name}.${name}_c_next_addition(rtl)
    port map (
        enable => c_next_addition_enable,
        clock  => c_next_addition_clock,
        x_1_address  => c_next_addition_x_1_address,
        x_2_address  => c_next_addition_x_2_address,
        y_address  => c_next_addition_y_address,
        x_1  => c_next_addition_x_1,
        x_2  => c_next_addition_x_2,
        y => c_next_addition_y,
        done  => c_next_addition_done
    );
    inst_${name}_fc_hamadard_product: entity ${work_library_name}.${name}_fc_hamadard_product(rtl)
    port map (
        enable => fc_hamadard_product_enable,
        clock  => fc_hamadard_product_clock,
        x_1_address  => fc_hamadard_product_x_1_address,
        x_2_address  => fc_hamadard_product_x_2_address,
        y_address  => fc_hamadard_product_y_address,
        x_1  => fc_hamadard_product_x_1,
        x_2  => fc_hamadard_product_x_2,
        y => fc_hamadard_product_y,
        done  => fc_hamadard_product_done
    );
    inst_${name}_ic_hamadard_product: entity ${work_library_name}.${name}_ic_hamadard_product(rtl)
    port map (
        enable => ic_hamadard_product_enable,
        clock  => ic_hamadard_product_clock,
        x_1_address  => ic_hamadard_product_x_1_address,
        x_2_address  => ic_hamadard_product_x_2_address,
        y_address  => ic_hamadard_product_y_address,
        x_1  => ic_hamadard_product_x_1,
        x_2  => ic_hamadard_product_x_2,
        y => ic_hamadard_product_y,
        done  => ic_hamadard_product_done
    );
    inst_${name}_oc_hamadard_product: entity ${work_library_name}.${name}_oc_hamadard_product(rtl)
    port map (
        enable => oc_hamadard_product_enable,
        clock  => oc_hamadard_product_clock,
        x_1_address  => oc_hamadard_product_x_1_address,
        x_2_address  => oc_hamadard_product_x_2_address,
        y_address  => oc_hamadard_product_y_address,
        x_1  => oc_hamadard_product_x_1,
        x_2  => oc_hamadard_product_x_2,
        y => oc_hamadard_product_y,
        done  => oc_hamadard_product_done
    );

end architecture;
