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
        IN_CHANNELS : integer := ${in_channels};
        OUT_CHANNELS : integer := ${out_channels};
        IN_SEQ_LEN : integer := ${seq_len};
        KERNEL_SIZE : integer := ${kernel_size};
        PADDING_LEN : integer := ${padding_len};
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
        enable: in std_logic;
        clock: in std_logic;
        x_address: out std_logic_vector(X_ADDR_WIDTH-1 downto 0);
        x: in std_logic_vector(DATA_WIDTH-1 downto 0);
        y_address: in std_logic_vector(Y_ADDR_WIDTH-1 downto 0);
        y: out std_logic_vector(DATA_WIDTH-1 downto 0);
        done: out std_logic
    );
end entity ${name};
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
    function multiply_accumulate(w : in signed(DATA_WIDTH downto 0);
                                x_in : in signed(DATA_WIDTH downto 0);
                                y_out : in signed(2*(DATA_WIDTH+1)-1 downto 0)
            ) return signed is
        variable temp : signed(2*(DATA_WIDTH+1)-1 downto 0) := (others=>'0');
    begin
        temp := w * x_in;
        return temp + y_out;
    end function;
    function scaling(x_to_scale : in signed(2 * (DATA_WIDTH + 1) - 1 downto 0);
    scaler_m : in signed(M_Q_DATA_WIDTH -1 downto 0);
    scaler_m_shift : in integer
    ) return signed is
    variable TMP_1 : signed(2 * (DATA_WIDTH + 1) + M_Q_DATA_WIDTH -1 downto 0) := (others=>'0');
    variable TMP_2 : signed(2 * (DATA_WIDTH + 1) + M_Q_DATA_WIDTH -1 downto 0) := (others=>'0');
    variable TMP_3 : signed(2 * (DATA_WIDTH + 1) + M_Q_DATA_WIDTH -1 downto 0) := (others=>'0');
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
    constant W_ADDR_WIDTH : integer :=  log2(IN_CHANNELS * KERNEL_SIZE * OUT_CHANNELS);
    constant B_ADDR_WIDTH : integer :=  log2(OUT_CHANNELS);
    constant OUT_SEQ_LEN : integer := IN_SEQ_LEN;
    signal M_Q_SIGNED : signed(M_Q_DATA_WIDTH - 1 downto 0) := to_signed(M_Q, M_Q_DATA_WIDTH);
    type t_layer_state is (s_stop, s_forward, s_finished);
    signal layer_state : t_layer_state;
    type t_mac_state is (s_stop, s_init, s_preload_b, s_preload, s_accumulate, s_scaling, s_output, s_done);
    signal mac_state : t_mac_state;
    signal s_x_addr : std_logic_vector(X_ADDR_WIDTH-1 downto 0) := (others=>'0');
    signal s_w_addr : std_logic_vector(W_ADDR_WIDTH-1 downto 0) := (others=>'0');
    signal s_w_std : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal s_x, s_w : signed(DATA_WIDTH-1 downto 0);
    signal x_sub_z : signed(DATA_WIDTH downto 0);
    signal w_sub_z : signed(DATA_WIDTH downto 0);
    signal macc_sum : signed((((DATA_WIDTH + 1) + (DATA_WIDTH + 1)) - 1) downto 0) := (others=>'0');
    signal s_b_addr : std_logic_vector(B_ADDR_WIDTH-1 downto 0) := (others=>'0');
    signal s_b_std : std_logic_vector(2 * (DATA_WIDTH + 1) - 1 downto 0);
    signal s_b : signed(2 * (DATA_WIDTH + 1) - 1 downto 0);
    signal y_scaled : signed(DATA_WIDTH downto 0) := (others=>'0');
    signal y_store_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal y_store_addr : unsigned(Y_ADDR_WIDTH-1 downto 0);
    signal y_store_addr_std : std_logic_vector(Y_ADDR_WIDTH-1 downto 0);
    signal y_store_en : std_logic;
    signal zero_padding_en : std_logic := '1';
begin
    done <= '1' when layer_state = s_finished else '0';
    s_x <= to_signed(Z_X+Z_X, s_x'length) when zero_padding_en = '1' else signed(x);
    FSM_PROC : process(clock, enable)
    begin
        if rising_edge(clock) then
            if enable = '0' then
                layer_state <= s_stop;
            else
                case layer_state is
                    when s_stop =>
                        layer_state <= s_forward;
                    when s_forward =>
                        if mac_state = s_done then
                            layer_state <= s_finished;
                        else
                            layer_state <= s_forward;
                        end if;
                    when s_finished =>
                        layer_state <= s_finished;
                end case;
            end if;
        end if;
    end process;
    MAIN_PROC : process(clock, layer_state)
        variable offset_kernel_weight, weight_idx : integer range 0 to IN_CHANNELS * KERNEL_SIZE * OUT_CHANNELS := 0;
        variable offset_x_idx, x_idx : integer range 0-PADDING_LEN to IN_CHANNELS * IN_SEQ_LEN + PADDING_LEN+2 := 0;
        variable cnt_in_kernel : integer range 0 to KERNEL_SIZE := 0;
        variable cnt_in_row : integer range 0 to IN_SEQ_LEN := 0;
        variable in_cnt_channel : integer range 0 to IN_CHANNELS := 0;
        variable out_cnt_channel : integer range 0 to OUT_CHANNELS := 0;
        variable y_idx : integer range 0 to OUT_CHANNELS * OUT_SEQ_LEN := 0;
        variable var_b_add_z_b : integer range -2**(2*(DATA_WIDTH+1)-1) to 2**(2*(DATA_WIDTH+1)-1)-1 := 0;
        variable var_y_store : signed(DATA_WIDTH downto 0);
        variable x_idx_addr : integer range 0 to IN_CHANNELS * IN_SEQ_LEN-1 := 0;
    begin
        if rising_edge(clock) then
            if layer_state = s_stop then
                mac_state <= s_init;
                offset_kernel_weight := 0;
                offset_x_idx := 0;
                weight_idx := 0;
                y_store_en <= '0';
                cnt_in_row := 0;
                in_cnt_channel := 0;
                out_cnt_channel := 0;
                zero_padding_en <= '1';
            else
                case mac_state is
                    when s_init =>
                        cnt_in_kernel := 0;
                        weight_idx := offset_kernel_weight;
                        x_idx := offset_x_idx - PADDING_LEN;
                        mac_state <= s_preload_b;
                    when s_preload_b =>
                        var_b_add_z_b := to_integer(s_b) + Z_B;
                        macc_sum <= to_signed(var_b_add_z_b, macc_sum'length);
                        mac_state <= s_preload;
                    when s_preload =>
                        mac_state <= s_accumulate;
                        x_sub_z <= to_signed(0, x_sub_z'length);
                        w_sub_z <= to_signed(0, w_sub_z'length);
                        weight_idx := weight_idx + 1;
                        x_idx := x_idx + 1;
                    when s_accumulate =>
                        y_store_en <= '0';
                        x_sub_z <= s_x - to_signed(Z_X, x_sub_z'length);
                        w_sub_z <= s_w - to_signed(Z_W, w_sub_z'length);
                        macc_sum <= multiply_accumulate(w_sub_z, x_sub_z, macc_sum);
                        if cnt_in_kernel < KERNEL_SIZE then
                            cnt_in_kernel := cnt_in_kernel + 1;
                            if cnt_in_kernel < KERNEL_SIZE-1 then
                                weight_idx := weight_idx + 1;
                                x_idx := x_idx + 1;
                            end if;
                            mac_state <= s_accumulate;
                        else
                            if in_cnt_channel < IN_CHANNELS-1 then
                                in_cnt_channel := in_cnt_channel + 1;
                                cnt_in_kernel := 0;
                                x_idx := x_idx + IN_SEQ_LEN-2;
                                weight_idx := weight_idx+1;
                                mac_state <= s_preload;
                            else
                                mac_state <= s_scaling;
                                weight_idx := offset_kernel_weight;
                                cnt_in_kernel := 0;
                                in_cnt_channel := 0;
                            end if;
                        end if;
                    when s_scaling =>
                        y_scaled <= scaling(macc_sum, M_Q_SIGNED, M_Q_SHIFT);
                        mac_state <= s_output;
                    when s_output =>
                        var_y_store := y_scaled + to_signed(Z_Y, y_scaled'length);
                        y_store_data <= std_logic_vector(resize(var_y_store, y_store_data'length));
                        y_store_en <= '1';
                        y_store_addr <= to_unsigned(y_idx, y_store_addr'length);
                        y_idx := y_idx + 1;
                        if cnt_in_row < OUT_SEQ_LEN-1 then
                            cnt_in_row := cnt_in_row + 1;
                            offset_x_idx := offset_x_idx + 1;
                            mac_state <= s_init;
                        else
                            cnt_in_row := 0;
                            offset_x_idx := 0;
                            offset_kernel_weight := offset_kernel_weight + KERNEL_SIZE * IN_CHANNELS;
                            out_cnt_channel := out_cnt_channel + 1;
                            if out_cnt_channel < OUT_CHANNELS then
                                mac_state <= s_init;
                            else
                                mac_state <= s_done;
                            end if;
                        end if;
                        weight_idx := offset_kernel_weight;
                    when s_done =>
                        y_store_en <= '0';
                        mac_state <= s_done;
                    when others =>
                        mac_state <= s_done;
                end case;
                s_w_addr <= std_logic_vector(to_unsigned(weight_idx, s_w_addr'length));
                if x_idx >= 0 and x_idx < IN_CHANNELS * IN_SEQ_LEN then
                    x_idx_addr := x_idx;
                    s_x_addr <= std_logic_vector(to_unsigned(x_idx_addr, s_x_addr'length));
                end if;
                if out_cnt_channel >= 0 and out_cnt_channel < OUT_CHANNELS then
                    s_b_addr <= std_logic_vector(to_unsigned(out_cnt_channel, s_b_addr'length));
                end if;
                if (cnt_in_row = 0 and cnt_in_kernel < PADDING_LEN) or (cnt_in_row=OUT_SEQ_LEN-1 and cnt_in_kernel >= KERNEL_SIZE-PADDING_LEN) then
                    zero_padding_en <= '1';
                else
                    zero_padding_en <= '0';
                end if;

            end if;
        end if;
    end process;
    x_address <= s_x_addr;
    y_store_addr_std <= std_logic_vector(y_store_addr);
    ram_y : entity ${work_library_name}.${name}_ram(rtl)
    generic map (
        RAM_WIDTH => DATA_WIDTH,
        RAM_DEPTH_WIDTH => Y_ADDR_WIDTH,
        RAM_PERFORMANCE => "LOW_LATENCY",
        RESOURCE_OPTION => Y_RESOURCE_OPTION,
        INIT_FILE => ""
    )
    port map  (
        addra  => y_store_addr_std,
        addrb  => y_address,
        dina   => y_store_data,
        clka   => clock,
        clkb   => clock,
        wea    => y_store_en,
        enb    => '1',
        rstb   => '0',
        regceb => '1',
        doutb  => y
    );
    rom_w : entity ${work_library_name}.${weights_rom_name}(rtl)
        port map (
            clk => clock,
            en => enable,
            addr => s_w_addr,
            data => s_w_std
        );
    s_w <= signed(s_w_std);
    rom_b : entity ${work_library_name}.${bias_rom_name}(rtl)
    port map (
        clk => clock,
        en => enable,
        addr => s_b_addr,
        data => s_b_std
    );
    s_b <= signed(s_b_std);
end architecture rtl;
