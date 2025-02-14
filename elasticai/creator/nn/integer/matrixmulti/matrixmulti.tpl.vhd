library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        IS_SCORE_MODE : boolean := ${is_score_mode};
        X_1_ADDR_WIDTH : integer := ${x_1_addr_width};
        X_2_ADDR_WIDTH : integer := ${x_2_addr_width};
        Y_ADDR_WIDTH : integer := ${y_addr_width};
        DATA_WIDTH : integer := ${data_width};
        X_1_DIM_A : integer := ${x_1_dim_a};
        X_1_DIM_B : integer := ${x_1_dim_b};
        X_1_DIM_C : integer := ${x_1_dim_c};
        X_2_DIM_A : integer := ${x_2_dim_a};
        X_2_DIM_B : integer := ${x_2_dim_b};
        X_2_DIM_C : integer := ${x_2_dim_c};
        Y_DIM_A : integer := ${y_dim_a};
        Y_DIM_B : integer := ${y_dim_b};
        Y_DIM_C : integer := ${y_dim_c};
        M_Q : integer := ${m_q};
        M_Q_SHIFT : integer := ${m_q_shift};
        Z_X_1 : integer := ${z_x_1};
        Z_X_2 : integer := ${z_x_2};
        Z_Y : integer := ${z_y};
        M_Q_DATA_WIDTH : integer := ${m_q_data_width};
        Y_RESOURCE_OPTION : string := "${resource_option}"
    );
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x_1_address : out std_logic_vector(X_1_ADDR_WIDTH-1 downto 0);
        x_2_address : out std_logic_vector(X_2_ADDR_WIDTH-1 downto 0);
        y_address : in std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
        x_1 : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        x_2 : in std_logic_vector(DATA_WIDTH - 1 downto 0);
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
    constant HALF_SIZE_X_1 : integer := X_1_DIM_A * X_1_DIM_B * X_1_DIM_C / 2;
    constant HALF_SIZE_X_2 : integer := X_2_DIM_A * X_2_DIM_B * X_2_DIM_C / 2;
    signal M_Q_SIGNED:signed(M_Q_DATA_WIDTH - 1 downto 0) := to_signed(M_Q, M_Q_DATA_WIDTH);
    signal n_clock : std_logic;
    signal reset : std_logic := '0';
    type t_layer_state is (s_stop, s_forward, s_finished);
    signal layer_state : t_layer_state;
    type t_mac_state is (s_stop, s_init, s_preload, s_accumulate, s_scaling, s_output, s_done);
    signal mac_state : t_mac_state;
    signal x_1_int: signed(DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal x_1_sub_z : signed(DATA_WIDTH downto 0) := (others=>'0');
    signal x_2_int: signed(DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal x_2_sub_z : signed(DATA_WIDTH downto 0) := (others=>'0');
    signal y_store_en : std_logic;
    signal y_scaled : signed(DATA_WIDTH downto 0) := (others=>'0');
    signal y_store_addr : integer range 0 to Y_DIM_A * Y_DIM_B * Y_DIM_C;
    signal y_store_addr_std : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
    signal y_store_data : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal macc_sum : signed(2 * (DATA_WIDTH + 1)-1 downto 0) := (others=>'0');
begin
    n_clock <= not clock;
    x_1_int <= signed(x_1);
    x_2_int <= signed(x_2);
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
        variable y_dim_a_idx : integer range 0 to Y_DIM_A := 0;
        variable y_dim_b_idx : integer range 0 to Y_DIM_B := 0;
        variable y_dim_c_idx : integer range 0 to Y_DIM_C := 0;
        variable input_1_idx : integer  range 0 to X_1_DIM_A * X_1_DIM_B * X_1_DIM_C-1 := 0;
        variable input_2_idx : integer  range 0 to X_2_DIM_A * X_2_DIM_B * X_2_DIM_C-1 := 0;
        variable mac_cnt : integer range 0 to X_1_DIM_C+1 := 0;
        variable output_idx : integer  range 0 to Y_DIM_A * Y_DIM_B * Y_DIM_C-1 := 0;
        variable var_product : signed(DATA_WIDTH - 1 downto 0);
        variable var_y_store : signed(DATA_WIDTH downto 0);
        variable input_1_offset : integer;
        variable input_2_offset : integer;
    begin
        if rising_edge(clock) then
            if layer_state=s_stop then
                mac_state <= s_init;
                y_dim_a_idx := 0;
                y_dim_b_idx := 0;
                y_dim_c_idx := 0;
                output_idx := 0;
                input_1_offset := 0;
                input_2_offset := 0;
                y_store_en <= '0';
            elsif layer_state=s_forward then
                case mac_state is
                    when s_init =>
                        input_1_idx := input_1_offset + y_dim_b_idx * X_1_DIM_C;
                        if IS_SCORE_MODE then
                            input_2_idx := input_2_offset + y_dim_c_idx * X_2_DIM_C;
                        else
                            input_2_idx := input_2_offset + y_dim_c_idx;
                        end if;
                        mac_state <= s_preload;
                        y_store_en <= '0';
                    when s_preload =>
                        x_1_sub_z <= to_signed(0, x_1_sub_z'length);
                        x_2_sub_z <= to_signed(0, x_2_sub_z'length);
                        macc_sum  <= to_signed(0, macc_sum'length);
                        input_1_idx := input_1_idx + 1;
                        if IS_SCORE_MODE then
                            input_2_idx := input_2_idx + 1;
                        else
                            input_2_idx := input_2_idx + X_2_DIM_C;
                        end if;
                        mac_state <= s_accumulate;
                    when s_accumulate =>
                        x_1_sub_z <= x_1_int - to_signed(Z_X_1, x_1_sub_z'length);
                        x_2_sub_z <= x_2_int - to_signed(Z_X_2, x_2_sub_z'length);
                        macc_sum <= multiply_accumulate(x_1_sub_z, x_2_sub_z, macc_sum);
                        mac_cnt := mac_cnt + 1;
                        if mac_cnt<=X_1_DIM_C then
                            if mac_cnt<X_1_DIM_C-1 then
                                input_1_idx := input_1_idx + 1;
                                if IS_SCORE_MODE then
                                    input_2_idx := input_2_idx + 1;
                                else
                                    input_2_idx := input_2_idx + X_2_DIM_C;
                                end if;
                            end if;
                            mac_state <= s_accumulate;
                        else
                            mac_state <= s_scaling;
                            mac_cnt := 0;
                        end if;
                    when s_scaling =>
                        y_scaled <= scaling(macc_sum, M_Q_SIGNED, M_Q_SHIFT);
                        mac_state <= s_output;
                    when s_output =>
                        var_y_store := y_scaled + to_signed(Z_Y, y_scaled'length);
                        y_store_data <= std_logic_vector(resize(var_y_store, y_store_data'length));
                        y_store_addr <= output_idx;
                        y_dim_c_idx := y_dim_c_idx + 1;
                        y_store_en <= '1';
                        if y_dim_c_idx < Y_DIM_C then
                            output_idx := output_idx + 1;
                            mac_state <= s_init;
                        else
                            y_dim_c_idx := 0;
                            y_dim_b_idx := y_dim_b_idx + 1;
                            if y_dim_b_idx < Y_DIM_B then
                                output_idx := output_idx + 1;
                                mac_state <= s_init;
                            else
                                input_1_offset := input_1_offset + X_1_DIM_C;
                                if IS_SCORE_MODE then
                                    input_2_offset := input_2_offset + X_1_DIM_C;
                                else
                                    input_2_offset := 0;
                                end if;
                                y_dim_b_idx := 0;
                                y_dim_a_idx := y_dim_a_idx + 1;
                                if y_dim_a_idx < Y_DIM_A then
                                    output_idx := output_idx + 1;
                                    mac_state <= s_init;
                                else
                                    mac_state <= s_done;
                                end if;
                            end if;
                        end if;
                    when others =>
                        mac_state <= s_done;
                end case;
            else
                mac_state <= s_done;
                y_store_en <= '0';
            end if;
            x_1_address <= std_logic_vector(to_unsigned(input_1_idx, x_1_address'length));
            x_2_address <= std_logic_vector(to_unsigned(input_2_idx, x_2_address'length));
        end if;
    end process ;
    y_store_addr_std <= std_logic_vector(to_unsigned(y_store_addr, y_store_addr_std'length));
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
end architecture;
