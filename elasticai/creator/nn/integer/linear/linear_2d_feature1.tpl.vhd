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
        IN_FEATURES : integer := ${in_features};
        OUT_FEATURES : integer := ${out_features};
        M_Q : integer := ${m_q};
        M_Q_SHIFT : integer := ${m_q_shift};
        Z_X : integer := ${z_x};
        Z_W : integer := ${z_w};
        Z_B : integer := ${z_b};
        Z_Y : integer := ${z_y};
        M_Q_DATA_WIDTH : integer := ${m_q_data_width};
        Y_RESOURCE_OPTION : string := "${resource_option}"
    );
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x_address : out std_logic_vector(X_ADDR_WIDTH - 1 downto 0);
        x   : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        y_address : in std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
        y  : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        done   : out std_logic
    );
end ${name};
architecture rtl of ${name} is
    constant MACC_OUT_WIDTH : integer := 2 * (DATA_WIDTH + 1) + 1;
    function multiply_accumulate(
                    w : in signed(DATA_WIDTH downto 0);
                    x_in : in signed(DATA_WIDTH downto 0);
                    y_out : in signed(MACC_OUT_WIDTH - 1 downto 0)
            ) return signed is
        variable TMP : signed(2 * (DATA_WIDTH + 1) - 1 downto 0) := (others=>'0');
    begin
        TMP := w * x_in;
        return TMP + y_out;
    end function;
    function scaling(x_to_scale : in signed(MACC_OUT_WIDTH - 1 downto 0);
    scaler_m : in signed(M_Q_DATA_WIDTH -1 downto 0);
    scaler_m_shift : in integer
    ) return signed is
    variable TMP_1 : signed(MACC_OUT_WIDTH + M_Q_DATA_WIDTH -1 downto 0) := (others=>'0');
    variable TMP_2 : signed(MACC_OUT_WIDTH + M_Q_DATA_WIDTH -1 downto 0) := (others=>'0');
    variable TMP_3 : signed(MACC_OUT_WIDTH + M_Q_DATA_WIDTH -1 downto 0) := (others=>'0');
    variable is_negative : boolean := x_to_scale(x_to_scale'left) = '1';
    begin
        if is_negative then
            TMP_1 := -x_to_scale * scaler_m;
        else
            TMP_1 := x_to_scale * scaler_m;
        end if;
        TMP_2 := shift_right(TMP_1, scaler_m_shift);
        TMP_3 := TMP_2;
        if TMP_1(scaler_m_shift-1) = '1' then
            TMP_3 := TMP_2 + 1;
        end if;
        if is_negative then
            return -resize(TMP_3, DATA_WIDTH + 1);
        else
            return resize(TMP_3, DATA_WIDTH + 1);
        end if;
    end function;
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
    signal M_Q_SIGNED:signed(M_Q_DATA_WIDTH - 1 downto 0) := to_signed(M_Q, M_Q_DATA_WIDTH);
    signal n_clock : std_logic;
    signal reset : std_logic := '0';
    type t_layer_state is (s_stop, s_forward, s_finished);
    signal layer_state : t_layer_state;
    type t_mac_state is (s_stop, s_init, s_preload, s_sub, s_mac, s_scaling, s_output, s_done);
    signal mac_state : t_mac_state;
    signal x_int : signed(DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal x_sub_z : signed(DATA_WIDTH downto 0) := (others=>'0');
    signal w_in : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal w_addr : std_logic_vector(log2(IN_FEATURES*OUT_FEATURES) - 1 downto 0) := (others=>'0');
    signal w_int : signed(DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal w_sub_z : signed(DATA_WIDTH downto 0) := (others=>'0');
    signal b_in : std_logic_vector(2 * (DATA_WIDTH + 1) - 1 downto 0) := (others=>'0');
    signal b_addr : std_logic_vector(log2(OUT_FEATURES) - 1 downto 0) := (others=>'0');
    signal b_int : signed(2 * (DATA_WIDTH + 1) - 1 downto 0) := (others=>'0');
    signal y_scaled : signed(DATA_WIDTH downto 0) := (others=>'0');
    signal y_store_addr : integer range 0 to OUT_FEATURES * NUM_DIMENSIONS;
    signal y_store_data : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal macc_sum : signed(2 * (DATA_WIDTH + 1) - 1 downto 0) := (others=>'0');
begin
    n_clock <= not clock;
    w_int <= signed(w_in);
    x_int <= signed(x);
    b_int <= signed(b_in);
    reset <= not enable;
    fsm : process (clock, reset)
    begin
        if (reset = '1') then
            layer_state <= s_stop;
            done <= '0';
        elsif rising_edge(clock) then
            if layer_state=s_stop then
                layer_state <= s_forward;
            elsif layer_state=s_forward then
                if mac_state=s_done then
                    layer_state <= s_finished;
                else
                    layer_state <= s_forward;
                end if;
            else
                done <= '1';
                layer_state <= s_finished;
            end if;
        end if;
    end process fsm;
    mac : process( clock, layer_state )
        variable dimension_idx : integer range 0 to NUM_DIMENSIONS  := 0;
        variable neuron_idx : integer range 0 to OUT_FEATURES := 0;
        variable input_idx : integer  range 0 to IN_FEATURES * NUM_DIMENSIONS  := 0;
        variable weight_idx : integer range 0 to OUT_FEATURES * IN_FEATURES := 0;
        variable bias_idx : integer range 0 to OUT_FEATURES := 0;
        variable output_idx : integer  range 0 to OUT_FEATURES * NUM_DIMENSIONS := 0;
        variable mac_cnt : integer range 0 to IN_FEATURES+1 := 0;
        variable input_offset : integer;
        variable var_product : signed(DATA_WIDTH - 1 downto 0);
        variable var_b_add_z_b : integer;
        variable var_y_store : signed(DATA_WIDTH downto 0);
    begin
        if rising_edge(clock) then
            if layer_state=s_stop then
                mac_state <= s_init;
                dimension_idx := 0;
                neuron_idx := 0;
                input_idx := 0;
                weight_idx := 0;
                bias_idx := 0;
                output_idx := 0;
                input_offset :=0;
                mac_cnt :=0;
            elsif layer_state=s_forward then
                case mac_state is
                    when s_init =>
                        input_idx := input_offset;
                        mac_cnt :=0;
                        mac_state <= s_preload;
                    when s_preload =>
                        x_sub_z <= to_signed(0, x_sub_z'length);
                        w_sub_z <= to_signed(0, w_sub_z'length);
                        var_b_add_z_b := to_integer(b_int) +Z_B;
                        macc_sum <= to_signed(var_b_add_z_b, macc_sum'length);
                        mac_state <= s_sub;
                    when s_sub =>
                        x_sub_z <= x_int - to_signed(Z_X, x_sub_z'length);
                        w_sub_z <= w_int - to_signed(Z_W, w_sub_z'length);
                        mac_state <= s_mac;
                    when s_mac =>
                        macc_sum <= multiply_accumulate(w_sub_z, x_sub_z, macc_sum);
                        mac_state <= s_scaling;
                    when s_scaling =>
                        y_scaled <= scaling(macc_sum, M_Q_SIGNED, M_Q_SHIFT);
                        mac_state <= s_output;
                    when s_output =>
                        var_y_store := y_scaled + to_signed(Z_Y, y_scaled'length);
                        y_store_data <= std_logic_vector(resize(var_y_store, y_store_data'length));
                        y_store_addr <= output_idx;
                        if neuron_idx < OUT_FEATURES-1 then
                            neuron_idx := neuron_idx + 1;
                            weight_idx := weight_idx + 1;
                            bias_idx := bias_idx + 1;
                            mac_state <= s_init;
                            output_idx := output_idx + 1;
                        else
                            if dimension_idx < NUM_DIMENSIONS - 1 then
                                dimension_idx := dimension_idx + 1;
                                input_idx := 0 + dimension_idx * IN_FEATURES;
                                neuron_idx := 0;
                                weight_idx := 0;
                                bias_idx := 0;
                                output_idx := output_idx + 1;
                                input_offset := input_offset + IN_FEATURES;
                                mac_state <= s_init;
                            else
                                mac_state <= s_done;
                            end if;
                        end if;
                    when others =>
                        mac_state <= s_done;
                end case;
            else
                mac_state <= s_done;
            end if;
            x_addr <= std_logic_vector(to_unsigned(input_idx, x_addr'length));
            w_addr <= std_logic_vector(to_unsigned(weight_idx, w_addr'length));
            b_addr <= std_logic_vector(to_unsigned(bias_idx, b_addr'length));
        end if;
    end process ;
    process_y_ram : process (clock)
        variable address_int : integer;
        type t_y_array is array (0 to OUT_FEATURES * NUM_DIMENSIONS) of std_logic_vector(DATA_WIDTH - 1 downto 0);
        variable y_ram : t_y_array;
        attribute rom_style : string;
        attribute rom_style of y_ram : variable is "auto";
    begin
        if rising_edge(clock) then
            y_ram(y_store_addr) := y_store_data;
            address_int := to_integer(unsigned(y_addr));
            y_out <= y_ram(address_int);
        end if;
    end process process_y_ram;
    rom_w : entity ${work_library_name}.${weights_rom_name}(rtl)
    port map  (
        clk  => clock,
        en   => '1',
        addr => w_addr,
        data => w_in
    );
    rom_b : entity ${work_library_name}.${bias_rom_name}(rtl)
    port map  (
        clk  => clock,
        en   => '1',
        addr => b_addr,
        data => b_in
    );
end architecture;
