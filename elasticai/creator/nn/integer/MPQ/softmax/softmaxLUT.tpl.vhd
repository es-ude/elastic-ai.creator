library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
generic (
    X_ADDR_WIDTH : integer := ${x_addr_width};
    Y_ADDR_WIDTH : integer := ${y_addr_width};
    X_DATA_WIDTH : integer := ${x_data_width};
    Y_DATA_WIDTH : integer := ${y_data_width};
    DIM_A: integer := ${dim_a};
    DIM_B: integer := ${dim_b};
    DIM_C: integer := ${dim_c};
    NUMERATOR_LUT_OUT_DATA_WIDTH: integer := ${numberator_lut_out_data_width};
    DENOMINATOR_LUT_OUT_DATA_WIDTH: integer := ${denominator_lut_out_data_width};
    Z_X: integer := ${z_x};
    Z_T: integer := ${z_t};
    Z_Y: integer := ${z_y};
    Y_RESOURCE_OPTION : string := "${resource_option}"
);
port (
    enable : in std_logic;
    clock : in std_logic;
    x_address : out std_logic_vector(X_ADDR_WIDTH - 1 downto 0);
    x : in std_logic_vector(X_DATA_WIDTH - 1 downto 0);
    y_address : in std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
	y : out std_logic_vector(Y_DATA_WIDTH - 1 downto 0);
    done : out std_logic
);
end entity ${name};
architecture rtl of ${name} is
    constant NUM_ELEMENT : integer := DIM_A*DIM_B*DIM_C;
    signal reset : std_logic := '0';
    signal x_int : signed(X_DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal denominator_lut_x: std_logic_vector(X_DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal denominator_lut_x_int: signed(X_DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal tmp_numerators_addr : integer range 0 to DIM_C-1 := 0;
    signal numerator_lut_y: std_logic_vector(NUMERATOR_LUT_OUT_DATA_WIDTH-1 downto 0) := (others=>'0');
    signal numerator_lut_y_int: signed(NUMERATOR_LUT_OUT_DATA_WIDTH-1 downto 0) := (others=>'0');
    signal max_value_sub_z: signed(Y_DATA_WIDTH downto 0) := (others=>'0');
    signal denominator_lut_y: std_logic_vector(DENOMINATOR_LUT_OUT_DATA_WIDTH-1 downto 0) := (others=>'0');
    signal denominator_lut_y_int: signed(DENOMINATOR_LUT_OUT_DATA_WIDTH-1 downto 0) := (others=>'0');
    signal denominator_sum_s : signed(DENOMINATOR_LUT_OUT_DATA_WIDTH downto 0) := (others=>'0');
    signal division_request: std_logic := '0';
    signal division_done: std_logic := '0';
    signal denominator_done: std_logic := '0';
    type t_numerators_array is array (0 to DIM_C) of signed(NUMERATOR_LUT_OUT_DATA_WIDTH-1 downto 0);
    signal numerators_ram : t_numerators_array;
    signal y_store_en : std_logic;
    signal y_int : signed(Y_DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal y_store_addr : integer range 0 to NUM_ELEMENT-1 := 0;
    signal y_store_addr_std : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
    signal y_store_data : std_logic_vector(Y_DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal divider_request, divider_done : std_logic := '0';
    signal divider_quotient, divider_dividend, divider_divisor, divider_remainder: signed(NUMERATOR_LUT_OUT_DATA_WIDTH downto 0) := (others=>'0');
    signal divider_quotient_std, divider_remainder_std : std_logic_vector(NUMERATOR_LUT_OUT_DATA_WIDTH downto 0) := (others=>'0');
begin
    reset <= not enable;
    x_int <= signed(x);
    numerator_lut_y_int <= signed(numerator_lut_y);
    denominator_lut_y_int <= signed(denominator_lut_y);
    calculate_denominator: process(reset,clock)
        variable dim_a_idx : integer range 0 to DIM_A-1;
        variable dim_b_idx : integer range 0 to DIM_B-1;
        variable dim_c_idx : integer range 0 to DIM_C-1;
        variable input_idx : integer range 0 to NUM_ELEMENT-1;
        variable denominator_sum : signed(DENOMINATOR_LUT_OUT_DATA_WIDTH downto 0) := (others=>'0');
        type t_state is (s_find_max, s_lookup);
        variable state : t_state;
        variable max_value : signed(X_DATA_WIDTH - 1 downto 0);
        variable pre_fetch : integer range 0 to 6;
        variable input_offset : integer range 0 to NUM_ELEMENT-1;
        variable tmp: integer;
    begin
        if reset = '1' then
            dim_a_idx := 0;
            dim_b_idx := 0;
            dim_c_idx := 0;
            input_idx := 0;
            denominator_sum := (others=>'0');
            division_request <= '0';
            denominator_done <= '0';
            max_value := ('1',others=>'0');
            input_offset := 0;
            denominator_lut_x_int <= (others=>'0');
            denominator_sum_s <= (others=>'0');
            max_value_sub_z <= (others=>'0');
        elsif rising_edge(clock) then
                if division_done = '1' and denominator_done = '0'  then
                    division_request <= '0';
                    if state=s_find_max then
                        if pre_fetch = 0 then
                            pre_fetch := 1;
                        elsif pre_fetch = 1 then
                        if x_int > max_value then
                            max_value := x_int;
                        end if;
                        if input_idx < NUM_ELEMENT-1 then
                            input_idx := input_idx + 1;
                        end if;
                        if dim_c_idx < DIM_C-1 then
                            dim_c_idx := dim_c_idx + 1;
                                pre_fetch := 0;
                        else
                            dim_c_idx := 0;
                            state := s_lookup;
                            max_value_sub_z <= to_signed(to_integer(max_value) - Z_X, max_value_sub_z'length);
                            pre_fetch := 0;
                            denominator_sum := (others=>'0');
                            input_idx := input_offset;
                            end if;
                        end if;
                    else
                        if pre_fetch = 0 then
                            pre_fetch := 1;
                        elsif pre_fetch = 1 then
                            tmp := to_integer(x_int) - to_integer(max_value_sub_z);
                            denominator_lut_x <= std_logic_vector(to_signed(tmp, denominator_lut_x'length));
                            pre_fetch := 2;
                        elsif pre_fetch < 5 then
                            pre_fetch := pre_fetch+1;
                        else
                            denominator_sum := denominator_sum + denominator_lut_y_int - to_signed(Z_T, denominator_lut_y_int'length);
                            numerators_ram(dim_c_idx) <= numerator_lut_y_int;
                            if input_idx < NUM_ELEMENT-1 then
                                input_idx := input_idx + 1;
                            end if;
                            if dim_c_idx < DIM_C-1 then
                                dim_c_idx := dim_c_idx + 1;
                            else
                                input_offset := input_idx;
                                dim_c_idx := 0;
                                max_value := ('1',others=>'0');
                                state := s_find_max;
                                if dim_b_idx < DIM_B-1 then
                                    dim_b_idx := dim_b_idx + 1;
                                else
                                    dim_b_idx := 0;
                                    if dim_a_idx < DIM_A-1 then
                                        dim_a_idx := dim_a_idx + 1;
                                    else
                                        dim_a_idx := 0;
                                        denominator_done <= '1';
                                    end if;
                                end if;
                                denominator_sum_s <= denominator_sum;
                                division_request <= '1';
                            end if;
                            pre_fetch := 0;
                        end if;
                    end if;
                end if;
        end if;
        x_address <= std_logic_vector(to_unsigned(input_idx, x_address'length));
    end process calculate_denominator;
    do_division: process (reset,clock, division_request)
    variable output_idx : integer range 0 to NUM_ELEMENT-1;
    variable dim_c_idx : integer range 0 to DIM_C-1;
    type t_state is (s_idle, s_run, s_done);
    variable state : t_state;
    variable tmp,tmp2: integer;
    variable sum_var : signed(DENOMINATOR_LUT_OUT_DATA_WIDTH downto 0) := (others=>'0');
    variable delay : integer range 0 to 10 := 0;
    begin
        if reset = '1' then
            division_done <= '1';
            output_idx := 0;
            dim_c_idx := 0;
            state := s_idle;
            done <= '0';
            divider_request <= '0';
            delay := 2;
            y_store_en <= '0';
        elsif rising_edge(clock) then
            if state = s_idle then
                if division_request = '1' then
                    state := s_run;
                    division_done <= '0';
                    dim_c_idx := 0;
                    sum_var := (denominator_sum_s);
                    y_store_en <= '0';
                end if;
            elsif state = s_run then
                if divider_request='0' then
                    divider_request <= '1';
                    divider_dividend <= resize(numerators_ram(dim_c_idx), divider_dividend'length);
                    divider_divisor <=  resize(denominator_sum_s, divider_divisor'length);
                else
                    if delay>0 then
                        delay := delay - 1;
                    else
                        if divider_done = '0' then
                        else
                            tmp :=to_integer(numerators_ram(dim_c_idx)) / to_integer(sum_var)+ Z_Y;
                                tmp2 := to_integer(divider_quotient) + Z_Y;
                            y_store_data <= std_logic_vector(to_signed(tmp2,y_store_data'length));
                            y_store_addr <= output_idx;
                            y_store_en <= '1';
                            if output_idx < NUM_ELEMENT-1 then
                                output_idx := output_idx + 1;
                            end if;
                            if dim_c_idx < DIM_C-1 then
                                dim_c_idx := dim_c_idx + 1;
                            else
                                state := s_done;
                            end if;
                            divider_request <= '0';
                            delay := 2;
                        end if;
                    end if;
                end if;
            elsif state = s_done then
                division_done <= '1';
                if division_request = '0' and denominator_done ='0' then
                    state := s_idle;
                else
                    done <= '1';
                    y_store_en <= '0';
                end if;
            end if;
        end if;
    end process do_division;
    y_store_addr_std <= std_logic_vector(to_unsigned(y_store_addr, y_store_addr_std'length));
    ram_y : entity ${work_library_name}.${name}_ram(rtl)
    generic map (
        RAM_WIDTH => Y_DATA_WIDTH,
        RAM_DEPTH_WIDTH => Y_ADDR_WIDTH,
        RAM_PERFORMANCE => "LOW_LATENCY",
        RESOURCE_OPTION => Y_RESOURCE_OPTION,
        INIT_FILE => "" )
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
    inst_${name}_numerator: entity ${work_library_name}.${name}_numerator
    port map (
        enable => enable,
        clock => clock,
        x => denominator_lut_x,
        y => numerator_lut_y
    );
    inst_${name}_denominator: entity ${work_library_name}.${name}_denominator
    port map (
        enable => enable,
        clock => clock,
        x => denominator_lut_x,
        y => denominator_lut_y
    );
    inst_${divider_name}: entity ${work_library_name}.${divider_name}
    port map (
        clock => clock,
        enable => divider_request,
        divisor => STD_LOGIC_VECTOR(divider_divisor),
        dividend => STD_LOGIC_VECTOR(divider_dividend),
        quotient => divider_quotient_std,
        remainder => divider_remainder_std,
        done => divider_done
    );
    divider_quotient <= signed(divider_quotient_std);
    divider_remainder <= signed(divider_remainder_std);
end architecture;
