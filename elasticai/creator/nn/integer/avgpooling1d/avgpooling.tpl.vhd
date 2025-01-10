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
        IN_NUM_DIMENSIONS : integer := ${in_num_dimensions};
        OUT_NUM_DIMENSIONS : integer := ${out_num_dimensions};
        M_Q : integer := ${m_q};
        M_Q_SHIFT : integer := ${m_q_shift};
        Z_X : integer := ${z_x};
        Z_Y : integer := ${z_y};
        M_Q_DATA_WIDTH : integer := ${m_q_data_width};
        Y_RESOURCE_OPTION : string := "${resource_option}"
    );
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x_addr : out std_logic_vector(X_ADDR_WIDTH - 1 downto 0);
        x_in   : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        y_addr : in std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
        y_out  : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        done   : out std_logic
    );
end ${name};
architecture rtl of ${name} is
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
    type t_mac_state is (s_stop, s_init, s_preload, s_accumulate, s_scaling, s_output, s_done);
    signal mac_state : t_mac_state;
    signal x_int : signed(DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal x_sub_z : signed(DATA_WIDTH downto 0) := (others=>'0');
    signal y_store_en : std_logic;
    signal y_scaled : signed(DATA_WIDTH downto 0) := (others=>'0');
    signal y_store_addr : integer range 0 to OUT_FEATURES*OUT_NUM_DIMENSIONS;
    signal y_store_addr_std : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
    signal y_store_data : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal acc_sum : signed(2 * (DATA_WIDTH + 1)-1 downto 0) := (others=>'0');
begin
    n_clock <= not clock;
    x_int <= signed(x_in);
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
        variable dimension_idx : integer range 0 to IN_NUM_DIMENSIONS-1 := 0;
        variable neuron_idx : integer range 0 to OUT_FEATURES-1 := 0;
        variable input_idx : integer  range 0 to IN_FEATURES*IN_NUM_DIMENSIONS-1 := 0;
        variable mac_cnt : integer range 0 to IN_NUM_DIMENSIONS+1 := 0;
        variable output_idx : integer  range 0 to OUT_FEATURES*OUT_NUM_DIMENSIONS-1 := 0;
        variable var_y_store : signed(DATA_WIDTH downto 0);
        variable input_offset : integer;
    begin
        if rising_edge(clock) then
            if layer_state=s_stop then
                mac_state <= s_init;
                dimension_idx := 0;
                neuron_idx := 0;
                input_idx := 0;
                output_idx := 0;
                mac_cnt :=0;
                input_offset :=0;
                y_store_en <= '0';
            elsif layer_state=s_forward then
                case mac_state is
                    when s_init =>
                        input_idx := input_offset;
                        mac_cnt :=0;
                        mac_state <= s_preload;
                        y_store_en <= '0';
                    when s_preload =>
                        x_sub_z <= to_signed(0, x_sub_z'length);
                        acc_sum <= (OTHERS=>'0');
                        input_idx := input_idx + IN_FEATURES;
                        mac_state <= s_accumulate;
                    when s_accumulate =>
                        x_sub_z <= x_int - to_signed(Z_X, x_sub_z'length);
                        acc_sum <= acc_sum + x_sub_z;
                        mac_cnt := mac_cnt + 1;
                        if mac_cnt<=IN_NUM_DIMENSIONS then
                            if mac_cnt<IN_NUM_DIMENSIONS-1 then
                                input_idx := input_idx + IN_FEATURES;
                            end if;
                            mac_state <= s_accumulate;
                        else
                            mac_state <= s_scaling;
                        end if;
                    when s_scaling =>
                        y_scaled <= scaling(acc_sum, M_Q_SIGNED, M_Q_SHIFT);
                        mac_state <= s_output;
                    when s_output =>
                        var_y_store := y_scaled + to_signed(Z_Y, y_scaled'length);
                        y_store_data <= std_logic_vector(resize(var_y_store, y_store_data'length));
                        y_store_addr <= output_idx;
                        y_store_en <= '1';
                        if neuron_idx<OUT_FEATURES-1 then
                            neuron_idx := neuron_idx + 1;
                            mac_state <= s_init;
                            output_idx := output_idx + 1;
                            input_offset:= input_offset + 1;
                        else
                            mac_state <= s_done;
                        end if;

                    when others =>
                        mac_state <= s_done;
                end case;
            else
                mac_state <= s_done;
                y_store_en <= '0';
            end if;
            x_addr <= std_logic_vector(to_unsigned(input_idx, x_addr'length));
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
        addrb  => y_addr,
        dina   => y_store_data,
        clka   => clock,
        clkb   => clock,
        wea    => y_store_en,
        enb    => '1',
        rstb   => '0',
        regceb => '1',
        doutb  => y_out
    );
end architecture;
