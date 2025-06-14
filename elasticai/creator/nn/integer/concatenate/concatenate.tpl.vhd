library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        X_1_ADDR_WIDTH : integer := ${x_1_addr_width};
        X_2_ADDR_WIDTH : integer := ${x_2_addr_width};
        Y_ADDR_WIDTH : integer := ${y_addr_width};
        DATA_WIDTH : integer := ${data_width};
        X1_NUM_FEATURES : integer := ${x1_num_features};
        X2_NUM_FEATURES : integer := ${x2_num_features};
        NUM_DIMENSIONS : integer := ${num_dimensions};
        M_Q_1 : integer := ${m_q_1};
        M_Q_2 : integer := ${m_q_2};
        M_Q_1_SHIFT : integer := ${m_q_1_shift};
        M_Q_2_SHIFT : integer := ${m_q_2_shift};
        Z_X_1 : integer := ${z_x_1};
        Z_X_2 : integer := ${z_x_2};
        Z_Y : integer := ${z_y};
        M_Q_DATA_WIDTH : integer := ${m_q_data_width};
        Y_RESOURCE_OPTION : string := "${resource_option}"
    );
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x_1_address : out std_logic_vector(X_1_ADDR_WIDTH - 1 downto 0);
        x_2_address : out std_logic_vector(X_2_ADDR_WIDTH - 1 downto 0);
        y_address : in std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
        x_1 : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        x_2 : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        y  : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        done   : out std_logic
    );
end ${name};
architecture rtl of ${name} is
    function scaling(x_to_scale : in signed(DATA_WIDTH downto 0);
    scaler_m : in signed(M_Q_DATA_WIDTH -1 downto 0);
    scaler_m_shift : in integer
    ) return signed is
    variable TMP_1 : signed(DATA_WIDTH + M_Q_DATA_WIDTH downto 0) := (others=>'0');
    variable TMP_2 : signed(DATA_WIDTH + M_Q_DATA_WIDTH downto 0) := (others=>'0');
    variable TMP_3 : signed(DATA_WIDTH + M_Q_DATA_WIDTH downto 0) := (others=>'0');
    variable is_negative : boolean := x_to_scale(x_to_scale'left) = '1';
    begin
        if is_negative then
            TMP_1 := -x_to_scale * scaler_m;
        else
            TMP_1 := x_to_scale * scaler_m;
        end if;
        TMP_2 := shift_right(TMP_1, scaler_m_shift);
        TMP_3 := TMP_2;
        if scaler_m_shift<DATA_WIDTH + M_Q_DATA_WIDTH  then
            if TMP_1(scaler_m_shift-1) = '1' then
                TMP_3 := TMP_2 + 1;
            end if;
        end if;
        if is_negative then
            return -resize(TMP_3, DATA_WIDTH + 1);
        else
            return resize(TMP_3, DATA_WIDTH + 1);
        end if;
    end function;
    signal n_clock : std_logic;
    signal reset : std_logic := '0';
    constant MINUS_X1_Z : signed(DATA_WIDTH downto 0) := to_signed(-Z_X_1, DATA_WIDTH+1);
    constant MINUS_X2_Z : signed(DATA_WIDTH downto 0) := to_signed(-Z_X_2, DATA_WIDTH+1);
    signal M_Q_1_SIGNED:signed(M_Q_DATA_WIDTH - 1 downto 0) := to_signed(M_Q_1, M_Q_DATA_WIDTH);
    signal M_Q_2_SIGNED:signed(M_Q_DATA_WIDTH - 1 downto 0) := to_signed(M_Q_2, M_Q_DATA_WIDTH);
    type t_layer_state is (s_stop, s_forward, s_finished);
    signal layer_state : t_layer_state;
    type t_add_state is (s_stop, s_init, s_preload, s_sub, s_scaling, s_output, s_done);
    signal add_state : t_add_state;
    signal x_1_int : signed(DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal x_2_int : signed(DATA_WIDTH - 1 downto 0) := (others=>'0');

    signal x_sub_z : signed(DATA_WIDTH downto 0) := (others=>'0');
    signal x_scaled : signed(DATA_WIDTH downto 0) := (others=>'0');

    signal y_store_en : std_logic;
    signal y_store_addr : integer range 0 to (X1_NUM_FEATURES+X2_NUM_FEATURES) * NUM_DIMENSIONS;
    signal y_store_addr_std : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
    signal y_store_data : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal read_x1_otherthan_x2 : boolean := true;
    signal x_int : signed(DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal x_m_q_signed: signed(M_Q_DATA_WIDTH - 1 downto 0);
    signal x_m_q_shift: integer;

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
                if add_state=s_done then
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
    add : process( clock, layer_state )
        variable dimension_idx : integer  range 0 to NUM_DIMENSIONS - 1 := 0;
        variable x1_input_idx, x1_end_idx : integer  range 0 to X1_NUM_FEATURES * NUM_DIMENSIONS := 0;
        variable x2_input_idx, x2_end_idx : integer  range 0 to X2_NUM_FEATURES * NUM_DIMENSIONS := 0;
        variable output_idx : integer  range 0 to (X1_NUM_FEATURES + X2_NUM_FEATURES) * NUM_DIMENSIONS := 0;
        variable var_y_store : signed( DATA_WIDTH  downto 0);
    begin
        if rising_edge(clock) then
            if layer_state=s_stop then
                add_state <= s_init;
                x1_input_idx := 0;
                x2_input_idx := 0;
                x1_end_idx := X1_NUM_FEATURES;
                x2_end_idx := X2_NUM_FEATURES;
                output_idx := 0;
                y_store_en <= '0';
                read_x1_otherthan_x2 <= true;
            elsif layer_state=s_forward then
                case add_state is
                    when s_init =>
                        add_state <= s_preload;
                        y_store_en <= '0';

                    when s_preload =>
                        if read_x1_otherthan_x2 then
                            x1_input_idx := x1_input_idx + 1;
                            x_m_q_signed <= M_Q_1_SIGNED;
                            x_m_q_shift <= M_Q_1_SHIFT;
                            x_sub_z <= MINUS_X1_Z;
                            x_int <= x_1_int;
                        else
                            x2_input_idx := x2_input_idx + 1;
                            x_m_q_signed <= M_Q_2_SIGNED;
                            x_m_q_shift <= M_Q_2_SHIFT;
                            x_sub_z <= MINUS_X2_Z;
                            x_int <= x_2_int;
                        end if;

                        add_state <= s_sub;

                    when s_sub =>
                        x_sub_z <= x_sub_z + x_int;
                        add_state <= s_scaling;
                    when s_scaling =>
                        x_scaled <= scaling(x_sub_z, x_m_q_signed, x_m_q_shift);
                        add_state <= s_output;
                    when s_output =>
                        var_y_store := x_scaled + to_signed(Z_Y, x_scaled'length);
                        y_store_data <= std_logic_vector(resize(var_y_store, y_store_data'length));
                        y_store_addr <= output_idx;
                        y_store_en <= '1';

                        output_idx := output_idx + 1;
                        if x2_input_idx < x2_end_idx then
                            add_state <= s_preload;
                        else
                            if x2_end_idx < X2_NUM_FEATURES*NUM_DIMENSIONS then
                                x1_end_idx := x1_end_idx + X1_NUM_FEATURES;
                                x2_end_idx := x2_end_idx + X2_NUM_FEATURES;
                            else
                                add_state <= s_done;
                            end if;
                        end if;

                    when others =>
                        add_state <= s_done;
                end case;
            else
                add_state <= s_done;
                y_store_en <= '0';
            end if;
            if x1_input_idx < X1_NUM_FEATURES then
                x_1_address <= std_logic_vector(to_unsigned(x1_input_idx, x_1_address'length));
            end if;
            if x2_input_idx < X2_NUM_FEATURES then
                x_2_address <= std_logic_vector(to_unsigned(x2_input_idx, x_2_address'length));
            end if;
            read_x1_otherthan_x2 <= (x1_input_idx < x1_end_idx);
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
