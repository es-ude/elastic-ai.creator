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
        NUM_FEATURES : integer := ${num_features};
        NUM_DIMENSIONS : integer := ${num_dimensions};
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
        x_1_address : out std_logic_vector(X_ADDR_WIDTH - 1 downto 0);
        x_2_address : out std_logic_vector(X_ADDR_WIDTH - 1 downto 0);
        y_address : in std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
        x_1 : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        x_2 : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        y  : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        done   : out std_logic
    );
end ${name};
architecture rtl of ${name} is
    function multiply(
        a : in signed(DATA_WIDTH downto 0);
        b : in signed(DATA_WIDTH downto 0)
    ) return signed is
    variable TMP : signed(2 * (DATA_WIDTH + 1) - 1 downto 0) := (others=>'0');
    begin
    TMP := a * b;
    return TMP;
    end function; -- end of multiply
    function shift_with_rounding(
        product : in signed(2 * (DATA_WIDTH + 1) + M_Q_DATA_WIDTH - 1 downto 0);
    scaler_m_shift : in integer
    ) return signed is
        variable shifted : signed(2 * (DATA_WIDTH + 1) + M_Q_DATA_WIDTH - 1 downto 0);
        variable round_bit : std_logic;
    begin

        round_bit := product(scaler_m_shift - 1);
        shifted := shift_right(product, scaler_m_shift);
        if round_bit = '1' then
            shifted := shifted + 1;
        end if;

        return resize(shifted, DATA_WIDTH + 1);
    end function;
    signal n_clock : std_logic;
    signal reset : std_logic := '0';
    signal M_Q_SIGNED:signed(M_Q_DATA_WIDTH - 1 downto 0) := to_signed(M_Q, M_Q_DATA_WIDTH);
    type t_layer_state is (s_stop, s_forward, s_finished);
    signal layer_state : t_layer_state;
    type t_add_state is (s_stop, s_init, s_preload, s_sub, s_scaling_1, s_scaling_2, s_sum, s_output, s_done);
    signal add_state : t_add_state;
    signal x_1_int : signed(DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal x_1_sub_z : signed(DATA_WIDTH downto 0) := (others=>'0');
    signal x_2_int : signed(DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal x_2_sub_z : signed(DATA_WIDTH downto 0) := (others=>'0');
    signal y_store_en : std_logic;
    signal y_store_addr : integer range 0 to NUM_FEATURES * NUM_DIMENSIONS;
    signal y_store_addr_std : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
    signal y_store_data : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal product : signed(2 * (DATA_WIDTH + 1) - 1 downto 0) := (others=>'0');
    signal product_scaled : signed(DATA_WIDTH downto 0) := (others=>'0');
    constant PRODUCT_SCALING_WIDTH : integer := 2 * (DATA_WIDTH + 1) + M_Q_DATA_WIDTH;
    signal product_to_scaling : signed(PRODUCT_SCALING_WIDTH - 1 downto 0) := (others=>'0');
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
        variable input_idx : integer  range 0 to NUM_FEATURES * NUM_DIMENSIONS-1 := 0;
        variable output_idx : integer  range 0 to NUM_FEATURES * NUM_DIMENSIONS-1 := 0;
        variable var_y_store : signed(DATA_WIDTH downto 0);
    begin
        if rising_edge(clock) then
            if layer_state=s_stop then
                add_state <= s_init;
                input_idx := 0;
                output_idx := 0;
                y_store_en <= '0';
            elsif layer_state=s_forward then
                case add_state is
                    when s_init =>
                        add_state <= s_preload;
                        y_store_en <= '0';
                    when s_preload =>
                        -- product <= (OTHERS=>'0');
                        add_state <= s_sub;
                    when s_sub =>
                        x_1_sub_z <= x_1_int - to_signed(Z_X_1, x_1_sub_z'length);
                        x_2_sub_z <= x_2_int - to_signed(Z_X_2, x_2_sub_z'length);
                        add_state <= s_sum;
                    when s_sum =>
                        product <= multiply(x_1_sub_z, x_2_sub_z);
                        add_state <= s_scaling_1;
                    when s_scaling_1 =>
                        product_to_scaling <= product * M_Q_SIGNED;
                        add_state <= s_scaling_2;
                    when s_scaling_2 =>
                        product_scaled <= shift_with_rounding(product_to_scaling, M_Q_SHIFT);
                        add_state <= s_output;
                    when s_output =>
                        var_y_store := product_scaled + to_signed(Z_Y, product_scaled'length);
                        y_store_data <= std_logic_vector(resize(var_y_store, y_store_data'length));
                        y_store_addr <= output_idx;
                        y_store_en <= '1';
                        if input_idx < NUM_DIMENSIONS * NUM_FEATURES-1 then
                            input_idx := input_idx + 1;
                            output_idx := output_idx + 1;
                            add_state <= s_init;
                        else
                            add_state <= s_done;
                        end if;
                    when others =>
                        add_state <= s_done;
                end case;
            else
                add_state <= s_done;
                y_store_en <= '0';
            end if;
            x_1_address <= std_logic_vector(to_unsigned(input_idx, x_1_address'length));
            x_2_address <= std_logic_vector(to_unsigned(input_idx, x_2_address'length));
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
