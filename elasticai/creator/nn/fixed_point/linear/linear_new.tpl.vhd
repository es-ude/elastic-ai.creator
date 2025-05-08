library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use ieee.numeric_std.all;

library ${work_library_name};
use ${work_library_name}.all;

entity ${layer_name} is -- layer_name is for distinguish same type of layers (with various weights) in one module
    generic (
        DATA_WIDTH   : integer := ${data_width};
        FRAC_WIDTH   : integer := ${frac_width};
        X_ADDR_WIDTH : integer := ${x_addr_width};
        Y_ADDR_WIDTH : integer := ${y_addr_width};
        IN_FEATURE_NUM : integer := ${in_feature_num};
        OUT_FEATURE_NUM : integer := ${out_feature_num};
        RESOURCE_OPTION : string := ${resource_option} -- can be "distributed", "block", or  "auto"
    );
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x_address : out std_logic_vector(X_ADDR_WIDTH-1 downto 0);
        y_address : in std_logic_vector(Y_ADDR_WIDTH-1 downto 0);

        x   : in std_logic_vector(DATA_WIDTH-1 downto 0);
        y  : out std_logic_vector(DATA_WIDTH-1 downto 0);

        done   : out std_logic
    );
end ${layer_name};

architecture rtl of ${layer_name} is
    -----------------------------------------------------------
    -- Functions
    -----------------------------------------------------------
    function cut_down(x: in signed(2*DATA_WIDTH-1 downto 0))return signed is
        variable TEMP2 : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
        variable TEMP3 : signed(FRAC_WIDTH-1 downto 0) := (others=>'0');
    begin

        TEMP2 := x(DATA_WIDTH+FRAC_WIDTH-1 downto FRAC_WIDTH);
        TEMP3 := x(FRAC_WIDTH-1 downto 0);
        if TEMP2(DATA_WIDTH-1) = '1' and TEMP3 /= 0 then
            TEMP2 := TEMP2 + 1;
        end if;

        if x>0 and TEMP2<0 then
            TEMP2 := ('0', others => '1');
        elsif x<0 and TEMP2>0 then
            TEMP2 := ('1', others => '0');
        end if;
        return TEMP2;
    end function;

    -- Log2 function is for calculating the bitwidth of the address lines
    -- for bias and weights rom
    function log2(val : INTEGER) return natural is
        variable res : natural;
    begin
        for i in 1 to ${log2_max_value} loop
            if (val <= (2 ** i)) then
                res := i;
                exit;
            end if;
        end loop;
        return res;
    end function log2;

    -----------------------------------------------------------
    -- Signals
    -----------------------------------------------------------
    constant FXP_ZERO : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    constant FXP_ONE : signed(DATA_WIDTH-1 downto 0) := to_signed(2**FRAC_WIDTH,DATA_WIDTH);

    type t_m_state is (s_idle, s_buf, s_inference, s_done);
    signal m_state : t_m_state := s_idle;

    signal n_clock : std_logic;
    signal w_in : std_logic_vector(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal b_in : std_logic_vector(DATA_WIDTH-1 downto 0) := (others=>'0');

    signal addr_w : std_logic_vector(log2(IN_FEATURE_NUM*OUT_FEATURE_NUM)-1 downto 0) := (others=>'0');
    --signal addr_b : std_logic_vector((log2(OUT_FEATURE_NUM)-1) downto 0) := (others=>'0');
    signal addr_b : std_logic_vector(Y_ADDR_WIDTH-1 downto 0) := (others=>'0');

    -- simple solution for the output buffer
    type t_y_array is array (0 to OUT_FEATURE_NUM-1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal y_ram : t_y_array;
    attribute rom_style : string;
    attribute rom_style of y_ram : signal is RESOURCE_OPTION;

    -- Enable buffer process
    type t_b_state is (s_b_idle, s_b_read, s_b_buffer, s_b_done);
    signal b_state : t_b_state := s_b_idle;
    signal enable_buffer : std_logic;
    signal done_buffer  : std_logic;

    -- Input, weight and bias buffer
    type t_x_array is array (0 to IN_FEATURE_NUM-1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    type t_w_array is array (0 to IN_FEATURE_NUM*OUT_FEATURE_NUM-1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    type t_b_array is array (0 to OUT_FEATURE_NUM-1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    type t_b_of_array is array (0 to OUT_FEATURE_NUM-1) of signed(2*DATA_WIDTH-1 downto 0);
    signal x_buf : t_x_array;
    signal w_buf : t_w_array;
    signal b_buf : t_b_array;
    signal b_of_buf : t_b_of_array;
    signal add_buf : t_b_of_array;

    -- Inference
    type t_i_state is (s_i_idle, s_i_mult, s_i_acc, s_i_add, s_i_write);
    signal i_state : t_i_state := s_i_idle;
    signal enable_inference : std_logic := '0';
    signal done_inference : std_logic;

    -- Arrays for variables
    constant NUM_COLS : integer := OUT_FEATURE_NUM; -- Number of cols
    constant NUM_ROWS : integer := IN_FEATURE_NUM; -- Number of rows
    type signed_array is array (0 to NUM_COLS-1, 0 to NUM_ROWS-1) of signed(2*DATA_WIDTH-1 downto 0); -- NUM_ROWS+1 because we need NUM_ROWS of Temps plus 1 for the accumulation
    signal temp : signed_array;

    type sum_array is array(0 to NUM_COLS-1) of signed(2*DATA_WIDTH-1 downto 0);
    signal debug_sum : sum_array;

begin

    -- connecting signals to ports
    n_clock <= not clock;

    linear_main : process (clock)

    begin

        if rising_edge(clock) then

            case(m_state) is

                when s_idle =>
                    done <= '0';
                    enable_buffer <= '0';
                    enable_inference <= '0';
                    if enable = '1' then
                        m_state <= s_buf;
                        enable_buffer <= '1';
                    end if;

                when s_buf =>
                    if done_buffer = '1' then
                        enable_buffer <= '0';
                        m_state <= s_inference;
                    end if;

                when s_inference =>
                    enable_inference <= '1';
                    if done_inference = '1' then
                        enable_inference <= '0';
                        m_state <= s_done;
                    end if;

                when s_done =>
                    done <= '1';
                    if enable = '0' then
                        m_state <= s_idle;
                    end if;

            end case;

        end if;

    end process linear_main;

    inference_sm : process (clock)
    begin

        if rising_edge(clock) then

            case i_state is

                when s_i_idle =>
                    done_inference <= '0';
                    if enable_inference = '1' then
                        i_state <= s_i_mult;
                    end if;

                when s_i_mult =>
                    i_state <= s_i_acc;

                when s_i_acc =>
                    i_state <= s_i_add;

                when s_i_add =>
                    i_state <= s_i_write;

                when s_i_write =>
                    done_inference <= '1';
                    if enable_inference = '0' then
                        i_state <= s_i_idle;
                    end if;

            end case;

        end if;

    end process;

    -- Process for each Output
    linear_inference : process (clock)

        variable sum : sum_array;

        begin

        if rising_edge(clock) then
            case i_state is

                when s_i_idle =>
                    for i in 0 to NUM_COLS-1 loop
                        sum(i) := (others => '0');
                        for j in 0 to NUM_ROWS-1 loop
                            temp(i, j) <= (others => '0');
                        end loop;
                    end loop;

                when s_i_mult =>
                    for i in 0 to NUM_COLS-1 loop
                        for j in 0 to NUM_ROWS-1 loop
                            temp(i, j) <= signed(x_buf(j)) * signed(w_buf(i * NUM_ROWS + j));
                        end loop;
                    end loop;

                when s_i_acc =>
                    for i in 0 to NUM_COLS-1 loop
                        b_of_buf(i) <= resize(signed(b_buf(i)), 2*DATA_WIDTH) sll FRAC_WIDTH;
                        for j in 0 to NUM_ROWS-1 loop
                            sum(i) := sum(i) + temp(i, j);
                        end loop;
                    end loop;

                when s_i_add =>
                    debug_sum <= sum;
                    for i in 0 to NUM_COLS-1 loop
                        add_buf(i) <= sum(i) + signed(b_of_buf(i));
                    end loop;

                when s_i_write =>
                    for i in 0 to NUM_COLS-1 loop
                        y_ram(i) <= std_logic_vector(cut_down(add_buf(i)));
                    end loop;

            end case;

        end if;

    end process;


    buffering : process (clock)

        variable b_cnt : integer := 0;
        variable x_cnt : integer := 0;
        variable w_cnt : integer := 0;

    begin

        if rising_edge(clock) then

            case(b_state) is

                when s_b_idle =>
                    b_cnt := 0;
                    x_cnt := 0;
                    w_cnt := 0;
                    done_buffer <= '0';
                    if enable_buffer = '1' then
                        b_state <= s_b_read;
                    end if;

                when s_b_read =>
                    if x_cnt < IN_FEATURE_NUM then
                        x_address <= std_logic_vector(to_unsigned(x_cnt, x_address'length));
                    end if;
                    if w_cnt < IN_FEATURE_NUM*OUT_FEATURE_NUM then
                        addr_w <= std_logic_vector(to_unsigned(w_cnt, addr_w'length));
                    end if;
                    if b_cnt < OUT_FEATURE_NUM then
                        addr_b <= std_logic_vector(to_unsigned(b_cnt, addr_b'length));
                    end if;
                    b_state <= s_b_buffer;

                when s_b_buffer =>
                    if x_cnt < IN_FEATURE_NUM then
                        x_buf(x_cnt) <= x;
                    end if;
                    if w_cnt < IN_FEATURE_NUM*OUT_FEATURE_NUM then
                        w_buf(w_cnt) <= w_in;
                    end if;
                    if b_cnt < OUT_FEATURE_NUM then
                        b_buf(b_cnt) <= b_in;
                    end if;
                    b_state <= s_b_done;

                when s_b_done =>
                    -- Determine next state based on counters
                    if x_cnt = IN_FEATURE_NUM and w_cnt = IN_FEATURE_NUM * OUT_FEATURE_NUM and b_cnt = OUT_FEATURE_NUM then
                        done_buffer <= '1';
                        if enable_buffer = '0' then
                            b_state <= s_b_idle;
                        end if;
                    else
                        -- Increment the appropriate counter and return to read state
                        if x_cnt < IN_FEATURE_NUM then
                            x_cnt := x_cnt + 1;
                        elsif w_cnt < IN_FEATURE_NUM * OUT_FEATURE_NUM then
                            w_cnt := w_cnt + 1;
                        elsif b_cnt < OUT_FEATURE_NUM then
                            b_cnt := b_cnt + 1;
                        end if;

                        b_state <= s_b_read;
                    end if;

            end case;

        end if;

    end process;

    y_reading : process (clock)
    begin
        --if (state=s_idle) or (state=s_stop) then -- Welche states sind ahnbar?
            if falling_edge(clock) then
                -- After the layer in at idle mode, y is readable
                -- but it only update at the rising edge of the clock
                y <= y_ram(to_integer(unsigned(y_address)));
            end if;
        --end if;
    end process y_reading;

    -- Weights
    rom_w : entity ${work_library_name}.${weights_rom_name}(rtl)
    port map  (
        clk  => n_clock,
        en   => '1',
        addr => addr_w,
        data => w_in
    );

    -- Bias
    rom_b : entity ${work_library_name}.${bias_rom_name}(rtl)
    port map  (
        clk  => n_clock,
        en   => '1',
        addr => addr_b,
        data => b_in
    );

end rtl;
