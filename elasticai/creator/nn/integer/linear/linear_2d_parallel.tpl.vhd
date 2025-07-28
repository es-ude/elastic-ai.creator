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
        OUT_FEATURES : integer := ${out_features};
        M_Q : integer := ${m_q};
        M_Q_SHIFT : integer := ${m_q_shift};
        Z_X : integer := ${z_x};
        Z_W : integer := ${z_w};
        Z_B : integer := ${z_b};
        Z_Y : integer := ${z_y};
        M_Q_DATA_WIDTH : integer := ${m_q_data_width};
        Y_RESOURCE_OPTION : string := "${resource_option}";
        UNROLL_FACTOR : integer := ${unroll_factor}
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
    constant PRODUCT_SCALING_WIDTH : integer := MACC_OUT_WIDTH + M_Q_DATA_WIDTH;
    function ceil_div(a : integer; b : integer) return integer is
    begin
        return (a + b - 1) / b;
    end function;
    -- Shift and round, keep original width
    function shift_with_rounding_fullwidth(
        product : in signed(MACC_OUT_WIDTH + M_Q_DATA_WIDTH - 1 downto 0);
        scaler_m_shift : in integer
    ) return signed is
        variable shifted : signed(MACC_OUT_WIDTH + M_Q_DATA_WIDTH - 1 downto 0);
        variable round_bit : std_logic;
        variable temp_result : signed(MACC_OUT_WIDTH + M_Q_DATA_WIDTH - 1 downto 0);
    begin
        if scaler_m_shift > 0 and scaler_m_shift <= product'length then
            round_bit := product(scaler_m_shift - 1);
        else
            round_bit := '0';
        end if;
        shifted := shift_right(product, scaler_m_shift);
        if round_bit = '1' then
            temp_result := shifted + 1;
        else
            temp_result := shifted;
        end if;
        return temp_result;
    end function;

    -- Clamp to DATA_WIDTH+1 bits
    function clamp_signed(
        value : in signed(MACC_OUT_WIDTH + M_Q_DATA_WIDTH - 1 downto 0)
    ) return signed is
        variable result : signed(DATA_WIDTH downto 0);
        constant MAX_VAL : signed(DATA_WIDTH downto 0) := to_signed(2**(DATA_WIDTH) - 1, DATA_WIDTH + 1);
        constant MIN_VAL : signed(DATA_WIDTH downto 0) := to_signed(-(2**(DATA_WIDTH)), DATA_WIDTH + 1);
    begin
        if value > resize(MAX_VAL, value'length) then
            result := MAX_VAL;
        elsif value < resize(MIN_VAL, value'length) then
            result := MIN_VAL;
        else
            result := resize(value, DATA_WIDTH + 1);
        end if;
        return result;
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
    type t_mac_state is (s_stop, s_init, s_preload, s_accumulate, s_scaling_0, s_scaling_1, s_scaling_2, s_scaling_3, s_output, s_done);
    signal mac_state : t_mac_state;
    signal x_int : signed(DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal x_sub_z : signed(DATA_WIDTH downto 0) := (others=>'0');
    signal w_in : std_logic_vector(UNROLL_FACTOR*DATA_WIDTH-1 downto 0) := (others=>'0');
    signal w_addr : std_logic_vector(log2(IN_FEATURES*ceil_div(OUT_FEATURES, UNROLL_FACTOR)) - 1 downto 0) := (others=>'0');
    type w_int_array_t is array (0 to UNROLL_FACTOR-1) of signed(DATA_WIDTH-1 downto 0);
    signal w_int : w_int_array_t := (others=>(others=>'0'));
    type w_sub_z_array_t is array (0 to UNROLL_FACTOR-1) of signed(DATA_WIDTH downto 0);
    signal w_sub_z_array : w_sub_z_array_t := (others=>(others=>'0'));
    signal b_in : std_logic_vector(UNROLL_FACTOR * 2 * (DATA_WIDTH + 1) - 1 downto 0) := (others=>'0');
    signal b_addr : std_logic_vector(log2(ceil_div(OUT_FEATURES, UNROLL_FACTOR)) - 1 downto 0) := (others=>'0');
    type b_int_array is array (0 to UNROLL_FACTOR-1) of signed( 2 * (DATA_WIDTH + 1) - 1 downto 0);
    signal b_int : b_int_array := (others=>(others=>'0'));
    signal y_store_en : std_logic;
    type y_shifted_array is array (0 to UNROLL_FACTOR-1) of signed(MACC_OUT_WIDTH + M_Q_DATA_WIDTH - 1 downto 0);
    signal y_shifted : y_shifted_array := (others=>(others=>'0'));
    type y_scaled_array is array (0 to UNROLL_FACTOR-1) of signed(DATA_WIDTH downto 0);
    signal y_scaled : y_scaled_array := (others=>(others=>'0'));
    signal y_store_addr : integer range 0 to OUT_FEATURES * NUM_DIMENSIONS;
    signal y_store_addr_std : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
    type y_store_data_array is array (0 to UNROLL_FACTOR-1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal y_store_data : y_store_data_array := (others=>(others=>'0'));
    type macc_sum_array_t is array (0 to UNROLL_FACTOR-1) of signed(MACC_OUT_WIDTH - 1 downto 0);
    signal macc_sum : macc_sum_array_t := (others=>(others=>'0'));

    type macc_product_array_t is array (0 to UNROLL_FACTOR-1) of signed(2 * (DATA_WIDTH+1) - 1 downto 0);
    signal macc_product : macc_product_array_t := (others=>(others=>'0'));

    type product_to_scaling_array_t is array (0 to UNROLL_FACTOR-1) of signed(PRODUCT_SCALING_WIDTH - 1 downto 0);
    signal product_to_scaling : product_to_scaling_array_t := (others=>(others=>'0'));

    signal ram_addr : integer range 0 to OUT_FEATURES * NUM_DIMENSIONS;
    signal ram_data : std_logic_vector(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal ram_write_en : std_logic := '0';

begin
    n_clock <= not clock;

    -- W input parsing using generate statement
    gen_w_int: for i in 0 to UNROLL_FACTOR-1 generate
        w_int(i) <= signed(w_in((i+1)*DATA_WIDTH-1 downto i*DATA_WIDTH));
        b_int(i) <= signed(b_in((i+1)*2*(DATA_WIDTH + 1) - 1 downto i*2*(DATA_WIDTH + 1)));
    end generate gen_w_int;

    x_int <= signed(x);

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
        variable dimension_idx : integer range 0 to NUM_DIMENSIONS - 1 := 0;
        variable neuron_idx : integer range 0 to OUT_FEATURES-1 := 0;
        variable input_idx : integer  range 0 to IN_FEATURES * NUM_DIMENSIONS - 1 := 0;
        variable weight_idx : integer range 0 to OUT_FEATURES * IN_FEATURES-1 := 0;
        variable bias_idx : integer range 0 to OUT_FEATURES-1 := 0;
        variable output_idx : integer  range 0 to OUT_FEATURES * NUM_DIMENSIONS - 1 := 0;
        variable mac_cnt : integer range 0 to IN_FEATURES+1 := 0;
        variable input_offset : integer;
        variable var_product : signed(DATA_WIDTH - 1 downto 0);
        type var_b_add_z_b_array_t is array (0 to UNROLL_FACTOR-1) of integer;
        variable var_b_add_z_b : var_b_add_z_b_array_t;
        type var_y_store_array_t is array (0 to UNROLL_FACTOR-1) of signed(DATA_WIDTH downto 0);
        variable var_y_store_array : var_y_store_array_t := (others=>(others=>'0'));
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
                y_store_en <= '0';
            elsif layer_state=s_forward then
                case mac_state is
                    when s_init =>
                        input_idx := input_offset;
                        mac_cnt :=0;
                        mac_state <= s_preload;
                        y_store_en <= '0';
                    when s_preload =>
                        for i in 0 to UNROLL_FACTOR-1 loop
                            var_b_add_z_b(i) := to_integer(b_int(i)) + Z_B;
                            macc_sum(i) <= to_signed(var_b_add_z_b(i), macc_sum(i)'length);
                        end loop;
                        input_idx := input_idx + 1;
                        weight_idx := weight_idx + 1;
                        mac_state <= s_accumulate;
                        -- x_sub_z <= to_signed(0, x_sub_z'length); -- commenting this line avoids combinational logic by vivado
                        w_sub_z_array <= (others=>(others=>'0'));
                        macc_product <= (others=>(others=>'0'));
                    when s_accumulate =>
                        x_sub_z <= x_int - to_signed(Z_X, x_sub_z'length);

                        for i in 0 to UNROLL_FACTOR-1 loop
                            w_sub_z_array(i) <= w_int(i) - to_signed(Z_W, w_sub_z_array(i)'length);
                            macc_product(i) <= w_sub_z_array(i) * x_sub_z;
                            macc_sum(i) <=  macc_sum(i) + macc_product(i);
                        end loop;

                        mac_cnt := mac_cnt + 1;
                        if mac_cnt <= IN_FEATURES then
                            if mac_cnt < IN_FEATURES-1 then
                                input_idx := input_idx + 1;
                                weight_idx := weight_idx + 1;
                            end if;
                            mac_state <= s_accumulate;
                        else
                            mac_state <= s_scaling_0;
                        end if;
                    when s_scaling_0 =>
                        for i in 0 to UNROLL_FACTOR-1 loop
                            macc_sum(i) <=  macc_sum(i) + macc_product(i);
                        end loop;
                        mac_state <= s_scaling_1;
                    when s_scaling_1 =>
                        for i in 0 to UNROLL_FACTOR-1 loop
                            product_to_scaling(i) <= macc_sum(i) * M_Q_SIGNED;
                        end loop;
                        mac_state <= s_scaling_2;
                    when s_scaling_2 =>
                        for i in 0 to UNROLL_FACTOR-1 loop
                            y_shifted(i) <= shift_with_rounding_fullwidth(product_to_scaling(i), M_Q_SHIFT);
                        end loop;
                        mac_state <= s_scaling_3;
                    when s_scaling_3 =>
                        for i in 0 to UNROLL_FACTOR-1 loop
                            y_scaled(i) <= clamp_signed(y_shifted(i));
                        end loop;
                        mac_state <= s_output;
                    when s_output =>
                        for i in 0 to UNROLL_FACTOR-1 loop
                            var_y_store_array(i) := y_scaled(i)  + to_signed(Z_Y, var_y_store_array(i)'length);
                            y_store_data(i) <= std_logic_vector(resize(var_y_store_array(i), y_store_data(i)'length));
                        end loop;
                        y_store_addr <= output_idx;
                        y_store_en <= '1';
                        if neuron_idx < OUT_FEATURES-UNROLL_FACTOR then
                            neuron_idx := neuron_idx + UNROLL_FACTOR;
                            weight_idx := weight_idx + 1;
                            bias_idx := bias_idx + 1;
                            mac_state <= s_init;
                            output_idx := output_idx + UNROLL_FACTOR;
                        else
                            if dimension_idx < NUM_DIMENSIONS - 1 then
                                dimension_idx := dimension_idx + 1;
                                input_idx := 0 + dimension_idx * IN_FEATURES;
                                neuron_idx := 0;
                                weight_idx := 0;
                                bias_idx := 0;
                                output_idx := output_idx + UNROLL_FACTOR;
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
                y_store_en <= '0';
            end if;
            x_address <= std_logic_vector(to_unsigned(input_idx, x_address'length));
            w_addr <= std_logic_vector(to_unsigned(weight_idx, w_addr'length));
            b_addr <= std_logic_vector(to_unsigned(bias_idx, b_addr'length));
        end if;
    end process ;

    save : process( y_store_en, clock )
    variable y_store_en_prev : std_logic := '0';
    variable store_active : boolean := false;
    variable store_count : integer range 0 to UNROLL_FACTOR := 0;
    begin
        if rising_edge(clock) then
            -- Detect rising edge of y_store_en
            if y_store_en = '1' and y_store_en_prev = '0' then
                store_active := true;
                store_count := 0;
            end if;

            -- Store two values sequentially
            if store_active then
                if store_count < UNROLL_FACTOR then
                    ram_addr <= y_store_addr + store_count;  -- Sequential addresses
                    ram_data <= y_store_data(store_count);  -- Use the selected data
                    ram_write_en <= '1';               -- Enable RAM write
                    store_count := store_count + 1;
                else
                    store_active := false;             -- Pause after 2 values
                    store_count := 0;
                    ram_write_en <= '0';               -- Disable RAM write
                end if;
            else
                ram_write_en <= '0';                   -- Ensure RAM write is disabled
            end if;

            -- Update previous value for edge detection
            y_store_en_prev := y_store_en;
        end if;
    end process save;

    y_store_addr_std <= std_logic_vector(to_unsigned(ram_addr, y_store_addr_std'length));
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
        dina   => ram_data,
        clka   => clock,
        clkb   => clock,
        wea    => ram_write_en,
        enb    => '1',
        rstb   => '0',
        regceb => '1',
        doutb  => y
    );
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
