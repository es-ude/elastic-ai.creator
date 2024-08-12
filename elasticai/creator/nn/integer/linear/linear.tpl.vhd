library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.all;

-----------------------------------------------------------
entity ${name} is
    generic (
        X_ADDR_WIDTH : integer := ${x_addr_width};
        Y_ADDR_WIDTH : integer := ${y_addr_width};
        DATA_WIDTH : integer := ${data_width};
        IN_FEATURES : integer := ${in_features};
        OUT_FEATURES : integer := ${out_features};
        SCALER : integer := ${scaler};
        SHIFT : integer := ${shift};
        Z_X : integer := ${z_x};
        Z_W : integer := ${z_w};
        Z_B : integer := ${z_b};
        Z_Y : integer := ${z_y};
        SCLAER_DATA_WIDTH : integer := 16;
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
-----------------------------------------------------------

architecture rtl of ${name} is
    ---------------------MAC Function-----------------------
    function multiply_accumulate(
                    w : in signed(DATA_WIDTH downto 0);
                    x_in : in signed(DATA_WIDTH downto 0);
                    y_0 : in signed(2 * (DATA_WIDTH + 1) - 1 downto 0)
            ) return signed is
        variable TMP : signed(2 * (DATA_WIDTH + 1) - 1 downto 0) := (others=>'0');
    begin
        TMP := w * x_in;
        return TMP + y_0;
    end function;
    ------------------Scaling Function---------------------
    function scaling(x_in : in signed(2 * (DATA_WIDTH + 1) - 1 downto 0);
    scaler_m : in signed(SCLAER_DATA_WIDTH -1 downto 0);
    shift_n : in integer
    ) return signed is
    variable TMP_1 : signed(2 * (DATA_WIDTH + 1) + SCLAER_DATA_WIDTH -1 downto 0) := (others=>'0');
    variable TMP_2 : signed(2 * (DATA_WIDTH + 1) + SCLAER_DATA_WIDTH -1 downto 0) := (others=>'0');
    variable TMP_3 : signed(2 * (DATA_WIDTH + 1) + SCLAER_DATA_WIDTH -1 downto 0) := (others=>'0');
    variable is_negative : boolean := x_in(x_in'left) = '1';
    begin
        if is_negative then
            TMP_1 := -x_in * scaler_m;
        else
            TMP_1 := x_in * scaler_m;
        end if;
        TMP_2 := shift_right(TMP_1, shift_n);
        TMP_3 := TMP_2;
        if TMP_1(shift_n-1) = '1' then
            TMP_3 := TMP_2 + 1;
        end if;
        if is_negative then
            return -resize(TMP_3, DATA_WIDTH + 1);
        else
            return resize(TMP_3, DATA_WIDTH + 1);
        end if;
    end function;
    ---------------------Log2 Function-----------------------
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
    -----------------------Signals-----------------------
    signal SCALER_SIGNED:signed(SCLAER_DATA_WIDTH - 1 downto 0) := to_signed(SCALER, SCLAER_DATA_WIDTH);

    signal n_clock : std_logic;
    signal reset : std_logic := '0';

    type t_layer_state is (s_stop, s_forward, s_finished);
    signal layer_state : t_layer_state;
    type t_mac_state is (s_stop, s_init, s_preload, s_accumulate, s_scaling, s_output, s_done);
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

    signal y_store_en : std_logic;
    signal y_scaled : signed(DATA_WIDTH downto 0) := (others=>'0');
    signal y_store_addr : integer range 0 to OUT_FEATURES;
    signal y_store_addr_std : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
    signal y_store_data : std_logic_vector(DATA_WIDTH - 1 downto 0);

    signal macc_sum : signed(2 * (DATA_WIDTH + 1) - 1 downto 0) := (others=>'0');
-----------------------------------------------------------
begin
    -- connecting signals to ports
    n_clock <= not clock;
    w_int <= signed(w_in);
    x_int <= signed(x);
    b_int <= signed(b_in);

    -- connects ports
    reset <= not enable;

    -- state machine
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

    -- mac process
    mac : process( clock, layer_state )
        variable neuron_idx : integer range 0 to OUT_FEATURES-1 := 0;
        variable input_idx : integer  range 0 to IN_FEATURES - 1 := 0;
        variable weight_idx : integer range 0 to OUT_FEATURES * IN_FEATURES-1 := 0;
        variable bias_idx : integer range 0 to OUT_FEATURES-1 := 0;
        variable output_idx : integer  range 0 to OUT_FEATURES - 1 := 0;

        variable mac_cnt : integer range 0 to IN_FEATURES+1 := 0;
        variable input_offset : integer;
        variable var_product : signed(DATA_WIDTH - 1 downto 0);
        variable var_b_add_z_b : integer;
        variable var_y_store : signed(DATA_WIDTH downto 0);
    begin
        if rising_edge(clock) then
            if layer_state=s_stop then
                mac_state <= s_init;
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
                        x_sub_z <= to_signed(0, x_sub_z'length);
                        w_sub_z <= to_signed(0, w_sub_z'length);
                        var_b_add_z_b := to_integer(b_int) +Z_B;
                        macc_sum <= to_signed(var_b_add_z_b, macc_sum'length);
                        input_idx := input_idx + 1;
                        weight_idx := weight_idx + 1;
                        mac_state <= s_accumulate;
                    when s_accumulate =>
                        x_sub_z <= x_int - to_signed(Z_X, x_sub_z'length);
                        w_sub_z <= w_int - to_signed(Z_W, w_sub_z'length);
                        macc_sum <= multiply_accumulate(w_sub_z, x_sub_z, macc_sum);
                        mac_cnt := mac_cnt + 1;
                        if mac_cnt <= IN_FEATURES then
                            if mac_cnt < IN_FEATURES-1 then
                                input_idx := input_idx + 1;
                                weight_idx := weight_idx + 1;
                            end if;
                            mac_state <= s_accumulate;
                        else
                            mac_state <= s_scaling;
                        end if;
                    when s_scaling =>
                        y_scaled <= scaling(macc_sum, SCALER_SIGNED, SHIFT);
                        mac_state <= s_output;
                    when s_output =>
                        var_y_store := y_scaled + to_signed(Z_Y, y_scaled'length);
                        y_store_data <= std_logic_vector(resize(var_y_store, y_store_data'length));
                        y_store_addr <= output_idx;
                        y_store_en <= '1';
                        if neuron_idx < OUT_FEATURES-1 then
                            neuron_idx := neuron_idx + 1;
                            weight_idx := weight_idx + 1;
                            bias_idx := bias_idx + 1;
                            mac_state <= s_init;
                            output_idx := output_idx + 1;
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
            x_address <= std_logic_vector(to_unsigned(input_idx, x_address'length));
            w_addr <= std_logic_vector(to_unsigned(weight_idx, w_addr'length));
            b_addr <= std_logic_vector(to_unsigned(bias_idx, b_addr'length));
        end if;
    end process ;

    ---------------- RAM Y ----------------
    y_store_addr_std <= std_logic_vector(to_unsigned(y_store_addr, y_store_addr_std'length));

    ram_y : entity work.${name}_ram(rtl)
    generic map (
        RAM_WIDTH => DATA_WIDTH,
        RAM_DEPTH_WIDTH => Y_ADDR_WIDTH,
        RAM_PERFORMANCE => "LOW_LATENCY",
        RESOURCE_OPTION => Y_RESOURCE_OPTION,
        INIT_FILE => ""  -- so relative path also xil_defaultlibs for ghdl, this path is relative to the path of the makefile, e.g "data/xx.dat"
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
        regceb => '0',
        doutb  => y
    );
    ---------------- RAM Y ----------------

    ---------------- WEIGHTS ROM ----------------
    rom_w : entity work.${weights_rom_name}(rtl)
    port map  (
        clk  => clock,
        en   => '1',
        addr => w_addr,
        data => w_in
    );

    ---------------- BIAS ROM  ----------------
    rom_b : entity work.${bias_rom_name}(rtl)
    port map  (
        clk  => clock,
        en   => '1',
        addr => b_addr,
        data => b_in
    );
end architecture;