-- Sync LSTM Cell, and scalable

LIBRARY ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.lstm_common.all;

entity lstm_cell is
    generic (
        DATA_WIDTH  : integer := 8;    -- that fixed point data has 16bits
        FRAC_WIDTH  : integer := 4;     -- and 8bits is for the factional part

        INPUT_SIZE  : integer := 5;     -- same as input_size of the lstm_cell in PyTorch
        HIDDEN_SIZE : integer := 3;     -- same as hidden_size of the lstm_cell in PyTorch

        X_H_ADDR_WIDTH : integer := 3;  -- equals to ceil(log2(input_size+hidden_size))
        HIDDEN_ADDR_WIDTH  : integer := 2; -- equals to ceil(log2(input_size))
        W_ADDR_WIDTH : integer := 5     -- equals to ceil(log2((input_size+hidden_size)*hidden_size)
        );
    port (
        clock     : in std_logic;
        reset     : in std_logic;
        enable    : in std_logic;    -- start computing when it is '1'
        x_h_we    : in std_logic;       -- Write enable for set the x_t and h_t-1
        x_h_data  : in std_logic_vector(DATA_WIDTH-1 downto 0);
        x_h_addr  : in std_logic_vector(X_H_ADDR_WIDTH-1 downto 0);

        c_we      : in std_logic;
        c_data_in : in std_logic_vector(DATA_WIDTH-1 downto 0);
        c_addr_in : in std_logic_vector(HIDDEN_ADDR_WIDTH-1 downto 0);

        done      : out std_logic;

        h_out_en   : in std_logic;
        h_out_addr : in std_logic_vector(HIDDEN_ADDR_WIDTH-1 downto 0); -- each rising_edge update h_out_data
        h_out_data : out std_logic_vector(DATA_WIDTH-1 downto 0)  --  accordingly when h_out_en is high
    );
end lstm_cell;

architecture rtl of lstm_cell is

    constant VECTOR_LENGTH : integer := INPUT_SIZE + HIDDEN_SIZE;
    constant MATRIX_LENGHT : integer := (INPUT_SIZE + HIDDEN_SIZE) * HIDDEN_SIZE;

    signal n_clock : std_logic;

    -- The state machine controls the gate component to be reused over time
    type state_t is (s_gates, s_i, s_f, s_g, s_o, s_c, s_h,s_update, idle);
    signal cell_state : state_t;

    signal std_x_h_read_addr : std_logic_vector((X_H_ADDR_WIDTH-1) downto 0):= (others => '0');
    signal std_x_h_read_data : std_logic_vector((DATA_WIDTH-1) downto 0):= (others => '0');
    signal std_c_read_addr : std_logic_vector((HIDDEN_ADDR_WIDTH-1) downto 0):= (others => '0');
    signal std_c_read_data : std_logic_vector((DATA_WIDTH-1) downto 0):= (others => '0');
    signal x_h_read_data      : signed(DATA_WIDTH-1 downto 0):= (others => '0');
    signal c_read_data : signed((DATA_WIDTH-1) downto 0):= (others => '0');

    signal c_t_addr,h_t_addr,h_t_addr_update : unsigned(HIDDEN_ADDR_WIDTH-1 downto 0):= (others => '0');

    signal x_h_config_we, c_config_we : std_logic;
    signal c_config_addr : std_logic_vector((HIDDEN_ADDR_WIDTH-1) downto 0):= (others => '0');
    signal c_config_data : std_logic_vector((DATA_WIDTH-1) downto 0):= (others => '0');
    signal x_h_config_addr : std_logic_vector((X_H_ADDR_WIDTH-1) downto 0):= (others => '0');
    signal x_h_config_data : std_logic_vector((DATA_WIDTH-1) downto 0):= (others => '0');

    signal gates_out_valid : std_logic;
    signal gate_process_enable : std_logic;
    signal gate_process_done : std_logic;
    signal vector_hadamard_product_done: std_logic;
    signal i_gate_out, f_gate_out, g_gate_out, o_gate_out : signed((DATA_WIDTH-1) downto 0):= (others => '0');
    signal i_gate_act, f_gate_act, g_gate_act, o_gate_act : signed((DATA_WIDTH-1) downto 0):= (others => '0');
    signal new_c, new_c_act : signed((DATA_WIDTH-1) downto 0):= (others => '0');
    signal new_h : signed((DATA_WIDTH-1) downto 0):= (others => '0');

    signal idx_hidden_out : integer range 0 to HIDDEN_SIZE;

    signal test_matrix_idx_s : integer range 0 to MATRIX_LENGHT;
    signal test_hidden_idx_s  : integer range 0 to HIDDEN_SIZE;
    signal test_running_state : integer range 0 to 7;
    signal test_dot_sum_i: signed((DATA_WIDTH-1) downto 0):= (others => '0');

    signal w_i_data : signed((DATA_WIDTH-1) downto 0):= (others => '0');
    signal w_f_data : signed((DATA_WIDTH-1) downto 0):= (others => '0');
    signal w_g_data : signed((DATA_WIDTH-1) downto 0):= (others => '0');
    signal w_o_data : signed((DATA_WIDTH-1) downto 0):= (others => '0');
    signal std_wi_out : std_logic_vector((DATA_WIDTH-1) downto 0):= (others => '0');
    signal std_wf_out : std_logic_vector((DATA_WIDTH-1) downto 0):= (others => '0');
    signal std_wg_out : std_logic_vector((DATA_WIDTH-1) downto 0):= (others => '0');
    signal std_wo_out : std_logic_vector((DATA_WIDTH-1) downto 0):= (others => '0');
    signal std_w_read_addr : std_logic_vector((W_ADDR_WIDTH-1) downto 0):= (others => '0');

    signal b_i_data : signed((DATA_WIDTH-1) downto 0):= (others => '0');
    signal b_f_data : signed((DATA_WIDTH-1) downto 0):= (others => '0');
    signal b_g_data : signed((DATA_WIDTH-1) downto 0):= (others => '0');
    signal b_o_data : signed((DATA_WIDTH-1) downto 0):= (others => '0');
    signal std_bi_out : std_logic_vector((DATA_WIDTH-1) downto 0):= (others => '0');
    signal std_bf_out : std_logic_vector((DATA_WIDTH-1) downto 0):= (others => '0');
    signal std_bg_out : std_logic_vector((DATA_WIDTH-1) downto 0):= (others => '0');
    signal std_bo_out : std_logic_vector((DATA_WIDTH-1) downto 0):= (others => '0');
    signal std_b_read_addr : std_logic_vector((HIDDEN_ADDR_WIDTH-1) downto 0):= (others => '0');

    type BUFFER_ARRAY is array (0 to HIDDEN_SIZE-1) of signed(DATA_WIDTH-1 downto 0);
    signal buffer_c_t: BUFFER_ARRAY;
    signal buffer_h_t: BUFFER_ARRAY;
    signal clk_hadamard:std_logic:='0';

begin

n_clock <= not clock;

--------------------------------------------------------
-- for clock scaling basically
clk_hadamard <= clock;

--process(clock)
--variable cnt:unsigned(3 downto 0):="0000";
--begin
--    if rising_edge(clock) then
--            cnt := cnt+1;
--            clk_hadamard <= cnt(1);
--    end if;
--end process;
--------------------------------------------------------

-- weights
-- [Wii,Whi]
rom_wi : entity work.wi_rom(rtl)
port map  (
    clk  => n_clock,
    en   => '1', -- todo can be optimized
    addr => std_w_read_addr,
    data => std_wi_out
);
w_i_data <= signed(std_wi_out);

-- [Wif,Whf]
rom_wf : entity work.wf_rom(rtl)
port map  (
    clk  => n_clock,
    en   => '1', -- todo can be optimized
    addr => std_w_read_addr,
    data => std_wf_out
);
w_f_data <= signed(std_wf_out);

-- [Wig,Whg]
rom_wg : entity work.wg_rom(rtl)
port map  (
    clk  => n_clock,
    en   => '1', -- todo can be optimized
    addr => std_w_read_addr,
    data => std_wg_out
);
w_g_data <= signed(std_wg_out);

-- [Wif,Whf]
rom_wo : entity work.wo_rom(rtl)
port map  (
    clk  => n_clock,
    en   => '1', -- todo can be optimized
    addr => std_w_read_addr,
    data => std_wo_out
);
w_o_data <= signed(std_wo_out);

-- biases
-- [Bii+Bhi]
rom_bi : entity work.bi_rom(rtl)
port map  (
    clk  => n_clock,
    en   => '1', -- todo can be optimized
    addr => std_b_read_addr,
    data => std_bi_out
);
b_i_data <= signed(std_bi_out);

-- [Bif+Bhf]
rom_bf : entity work.bf_rom(rtl)
port map  (
    clk  => n_clock,
    en   => '1', -- todo can be optimized
    addr => std_b_read_addr,
    data => std_bf_out
);
b_f_data <= signed(std_bf_out);

-- [Big+Bhg]
rom_bg : entity work.bg_rom(rtl)
port map  (
    clk  => n_clock,
    en   => '1', -- todo can be optimized
    addr => std_b_read_addr,
    data => std_bg_out
);
b_g_data <= signed(std_bg_out);

-- [Bio+Bho]
rom_bo : entity work.bo_rom(rtl)
port map  (
    clk  => n_clock,
    en   => '1', -- todo can be optimized
    addr => std_b_read_addr,
    data => std_bo_out
);
b_o_data <= signed(std_bo_out);


-- Instantiation x input buffer
buffer_x_in : entity work.dual_port_2_clock_ram(rtl)
generic map (
    RAM_WIDTH => DATA_WIDTH,
    RAM_DEPTH_WIDTH => X_H_ADDR_WIDTH,
    RAM_PERFORMANCE => "LOW_LATENCY",
    INIT_FILE => "" -- so relative path also works for ghdl, this path is relative to the path of the makefile, e.g "data/xx.dat"
)
port map  (
    addra  => x_h_config_addr,
    addrb  => std_x_h_read_addr,
    dina   => x_h_config_data,
    clka   => clock,
    clkb   => n_clock,
    wea    => x_h_config_we,
    enb    => '1',
    rstb   => '0',
    regceb => '0',
    doutb  => std_x_h_read_data
);
x_h_read_data <= signed(std_x_h_read_data);

x_h_config_addr <= x_h_addr;
x_h_config_data <= x_h_data;
x_h_config_we <= x_h_we;

buffer_c_in : entity work.dual_port_2_clock_ram(rtl)
generic map (
    RAM_WIDTH => DATA_WIDTH,
    RAM_DEPTH_WIDTH => HIDDEN_ADDR_WIDTH,
    RAM_PERFORMANCE => "LOW_LATENCY",
    INIT_FILE => ""
)
port map  (
    addra  => c_config_addr,
    addrb  => std_c_read_addr,
    dina   => c_config_data,
    clka   => clock,
    clkb   => clk_hadamard,
    wea    => c_config_we,
    enb    => '1',
    rstb   => '0',
    regceb => '0',
    doutb  => std_c_read_data
);

std_c_read_addr <= std_logic_vector(c_t_addr);
c_read_data <= signed(std_c_read_data);

c_config_addr <= c_addr_in;
c_config_data <= c_data_in;
c_config_we <= c_we;


MAIN_PROCESS : process(clock, enable, reset)
begin
    if reset = '1' then
        cell_state <= s_gates;
        done <= '0';
        gate_process_enable <= '0';
    elsif enable = '1' then
        if rising_edge(clock) then
            if cell_state = s_gates then
                gate_process_enable <= '1';
                if vector_hadamard_product_done ='1' then
                    cell_state <= idle;
                end if;
            else
                done <= '1';
            end if;
        end if;
    end if;
end process;

gate_process : process(clock, gate_process_enable, reset)
    variable var_x_h_idx : integer range 0 to VECTOR_LENGTH;
    variable is_data_prefetched : std_logic;
    variable dot_sum_i, dot_sum_f,dot_sum_g, dot_sum_o : signed(DATA_WIDTH-1 downto 0);
    variable mul_i, mul_f, mul_g, mul_o : signed(DATA_WIDTH-1 downto 0);
    variable var_matrix_idx : integer range 0 to MATRIX_LENGHT;
    variable var_hidden_idx : integer range 0 to HIDDEN_SIZE;
    variable var_x_h_data : signed(DATA_WIDTH-1 downto 0);
    variable var_w_i_data : signed(DATA_WIDTH-1 downto 0);
    variable var_w_g_data : signed(DATA_WIDTH-1 downto 0);
    variable var_w_o_data : signed(DATA_WIDTH-1 downto 0);
    variable var_w_f_data : signed(DATA_WIDTH-1 downto 0);
begin
    if reset = '1' then

        var_x_h_idx := 0;
        var_matrix_idx := 0;
        var_hidden_idx := 0;
        gate_process_done <= '0';
        gates_out_valid <= '0';
        is_data_prefetched := '0';
        dot_sum_i := (others=>'0');
        dot_sum_f := (others=>'0');
        dot_sum_g := (others=>'0');
        dot_sum_o := (others=>'0');
    elsif rising_edge(clock) then

        if gate_process_enable = '1' and gate_process_done='0' then
            if is_data_prefetched ='0' then
                is_data_prefetched := '1';
                var_x_h_data := x_h_read_data;
                var_w_f_data := w_f_data;
                var_w_i_data := w_i_data;
                var_w_g_data := w_g_data;
                var_w_o_data := w_o_data;

            else

--                if var_x_h_idx = 1 then   -- need to change this depends on the clock of partB
                gates_out_valid <= '0';
--                end if;

                mul_i := multiply_16_8(var_x_h_data, var_w_i_data);
                mul_f := multiply_16_8(var_x_h_data, var_w_f_data);
                mul_g := multiply_16_8(var_x_h_data, var_w_g_data);
                mul_o := multiply_16_8(var_x_h_data, var_w_o_data);

                dot_sum_i := dot_sum_i + mul_i;
                dot_sum_f := dot_sum_f + mul_f;
                dot_sum_g := dot_sum_g + mul_g;
                dot_sum_o := dot_sum_o + mul_o;

                var_x_h_idx := var_x_h_idx + 1;
                var_matrix_idx := var_matrix_idx + 1;
                is_data_prefetched := '0';
                if var_x_h_idx = VECTOR_LENGTH then
                    gates_out_valid <= '1';
                    dot_sum_i := dot_sum_i+b_i_data;
                    dot_sum_f := dot_sum_f+b_f_data;
                    dot_sum_g := dot_sum_g+b_g_data;
                    dot_sum_o := dot_sum_o+b_o_data;

                    i_gate_out <= dot_sum_i;
                    f_gate_out <= dot_sum_f;
                    g_gate_out <= dot_sum_g;
                    o_gate_out <= dot_sum_o;

                    idx_hidden_out <= var_hidden_idx;
                    var_x_h_idx := 0;

                    var_hidden_idx := var_hidden_idx +1;
                    if var_hidden_idx=HIDDEN_SIZE then
                        -- finished all gates
                        gate_process_done <= '1';
                    else
                        dot_sum_i := (others=>'0');
                        dot_sum_f := (others=>'0');
                        dot_sum_g := (others=>'0');
                        dot_sum_o := (others=>'0');
                    end if;
                end if;
            end if;
        end if;
    end if;
    std_x_h_read_addr <= std_logic_vector(to_unsigned(var_x_h_idx, std_x_h_read_addr'length));
    std_w_read_addr <= std_logic_vector(to_unsigned(var_matrix_idx, std_w_read_addr'length));
    std_b_read_addr <= std_logic_vector(to_unsigned(var_hidden_idx, std_b_read_addr'length));

    test_matrix_idx_s <= var_matrix_idx;
    test_hidden_idx_s <= var_hidden_idx;
    test_dot_sum_i <= dot_sum_i;
end process;

I_SIGMOID: entity work.sigmoid(rtl)
port map (
    i_gate_out,
    i_gate_act
);

F_SIGMOID: entity work.sigmoid(rtl)
port map (
    f_gate_out,
    f_gate_act
);

O_SIGMOID: entity work.sigmoid(rtl)
port map (
    o_gate_out,
    o_gate_act
);

G_TANH: entity work.tanh(rtl)
port map (
    g_gate_out,
    g_gate_act
);

C_TANH: entity work.tanh(rtl)
port map (
    new_c,
    new_c_act
);

vector_hadamard_product: process(reset, clk_hadamard, gates_out_valid)
    variable var_running_state : integer range 0 to 7;
    variable mul_f_c, mul_i_g: signed(DATA_WIDTH-1 downto 0);
    variable var_new_c,var_new_h : signed(DATA_WIDTH-1 downto 0);
    variable var_buffer_c_t: BUFFER_ARRAY;
    variable var_buffer_h_t: BUFFER_ARRAY;
    variable new_c_idx, new_h_idx, update_cnt : integer range 0 to HIDDEN_SIZE;
    variable var_c_read_data, var_f_gate_act, var_i_gate_act, var_g_gate_act, var_o_gate_act : signed(DATA_WIDTH-1 downto 0);
    variable var_new_c_act : signed(DATA_WIDTH-1 downto 0);
begin

    if rising_edge(clk_hadamard) then
        if reset = '1' then
            var_running_state := 0;
            new_c_idx := 0;
            new_h_idx := 0;
            update_cnt := 0;
            vector_hadamard_product_done <='0';

        else

            if gates_out_valid='1' and var_running_state = 0 then
                var_running_state := 1;

            elsif var_running_state=1 then
                var_f_gate_act := f_gate_act;
                var_i_gate_act := i_gate_act;
                var_c_read_data := c_read_data;
                var_g_gate_act := g_gate_act;
                var_o_gate_act := o_gate_act;

                mul_f_c := multiply_16_8(var_f_gate_act, var_c_read_data);
                mul_i_g := multiply_16_8(var_i_gate_act,var_g_gate_act);
                var_new_c := mul_f_c + mul_i_g;
                new_c <= var_new_c;
                var_running_state := 2;
                var_buffer_c_t(new_c_idx) := var_new_c;
                new_c_idx := new_c_idx + 1;
            elsif var_running_state=2 then
                var_new_c_act := new_c_act;
                var_running_state:=3;
            elsif var_running_state=3 then
                var_new_h := multiply_16_8(var_o_gate_act, var_new_c_act);
                new_h <= var_new_h;
                var_buffer_h_t(new_h_idx) := var_new_h;
                new_h_idx := new_h_idx+1;

                if gate_process_done='0' then
                    var_running_state := 0;
                else
                    var_running_state := 4;
                    update_cnt := 0;
                end if;
            elsif var_running_state=4 then -- update the temp data to mem
                buffer_c_t(update_cnt) <= var_buffer_c_t(update_cnt);
                buffer_h_t(update_cnt) <= var_buffer_h_t(update_cnt);
                update_cnt := update_cnt+1;
                if update_cnt=HIDDEN_SIZE then
                    vector_hadamard_product_done <='1';
                    var_running_state := 5;
                end if;
            end if;
        end if;
    end if;
    c_t_addr <= to_unsigned(new_c_idx, c_t_addr'length);
    test_running_state <= var_running_state;
end process;


-- for reading out output
read_process: process(clock, h_out_en)
    variable var_addr_integer : integer range 0 to HIDDEN_SIZE-1;
begin
    if rising_edge(clock) and h_out_en='1' then
        var_addr_integer := to_integer(unsigned(h_out_addr));
        h_out_data <= std_logic_vector(buffer_h_t(var_addr_integer));
    end if;
end process;

end rtl;
