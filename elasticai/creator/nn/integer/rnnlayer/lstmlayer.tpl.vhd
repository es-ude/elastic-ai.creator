library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        DATA_WIDTH : integer := ${data_width};
        NUM_DIMENSIONS : integer := ${num_dimensions};
        X_1_ADDR_WIDTH : integer := ${x_1_addr_width};
        X_2_ADDR_WIDTH : integer := ${x_2_addr_width};
        X_3_ADDR_WIDTH : integer := ${x_3_addr_width};
        Y_1_ADDR_WIDTH : integer := ${y_1_addr_width};
        Y_2_ADDR_WIDTH : integer := ${y_2_addr_width};
        Y_3_ADDR_WIDTH : integer := ${y_3_addr_width};
        X_1_COUNT : integer := ${x_1_count};
        X_2_COUNT : integer := ${x_2_count};
        X_3_COUNT : integer := ${x_3_count};
        Y_1_COUNT : integer := ${y_1_count};
        Y_2_COUNT : integer := ${y_2_count};
        Y_3_COUNT : integer := ${y_3_count};
        RESOURCE_OPTION : string := "${resource_option}"
    );
    port
    (
        enable : in std_logic;
        clock : in std_logic;
        x_1_address : out std_logic_vector(X_1_ADDR_WIDTH - 1 downto 0);
        x_2_address : out std_logic_vector(X_2_ADDR_WIDTH - 1 downto 0);
        x_3_address : out std_logic_vector(X_3_ADDR_WIDTH - 1 downto 0);
        y_1_address : in std_logic_vector(Y_1_ADDR_WIDTH - 1 downto 0);
        y_2_address : in std_logic_vector(Y_2_ADDR_WIDTH - 1 downto 0);
        y_3_address : in std_logic_vector(Y_2_ADDR_WIDTH - 1 downto 0);
        x_1 : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        x_2 : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        x_3 : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        y_1 : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        y_2 : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        y_3 : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        done : out std_logic
    );
end ${name};
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
    signal lstm_cell_enable : std_logic;
    signal lstm_cell_clock : std_logic;
    signal lstm_cell_x_1_address : std_logic_vector(log2(NUM_DIMENSIONS) - 1 downto 0);
    signal lstm_cell_x_2_address : std_logic_vector(log2(X_2_COUNT) - 1 downto 0);
    signal lstm_cell_x_3_address : std_logic_vector(log2(X_3_COUNT) - 1 downto 0);
    signal lstm_cell_x_1_data: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal lstm_cell_x_2_data: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal lstm_cell_x_3_data: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal lstm_cell_y_1_address, temp_addr : std_logic_vector(log2(Y_2_COUNT) - 1 downto 0);
    signal lstm_cell_y_2_address : std_logic_vector(log2(Y_3_COUNT) - 1 downto 0);
    signal lstm_cell_y_1_data: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal lstm_cell_y_2_data: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal lstm_cell_done: std_logic;
    signal read_states_from_prev_iter : boolean := false;
    type t_cell_state is (s_stop, s_start, s_wait, s_read_out, s_done);
    signal cell_state : t_cell_state;
    signal loop_counter : integer range 0 to X_1_COUNT := 0;
    signal reset : std_logic;
    signal lstm_cell_x_1_address_offset : integer range 0 to X_1_COUNT := 0;
    signal read_out_done : std_logic;
    signal y1_rd_out_data, cell_y1_store_data : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal y1_rd_out_addr : std_logic_vector(Y_1_ADDR_WIDTH - 1 downto 0);
    signal x_1_addr_int : integer range 0 to X_1_COUNT := 0;
    signal cell_y1_store_en : std_logic:='0';
    signal cell_y1_store_addr_std : std_logic_vector(Y_1_ADDR_WIDTH - 1 downto 0);
    signal cell_y1_read_addr : std_logic_vector(Y_2_ADDR_WIDTH - 1 downto 0); -- this is hidden states address, so maps to y_2_address
begin
    reset <= not enable;
    lstm_cell_clock <= clock;
    read_states_from_prev_iter <= (loop_counter > 0);
    x_1_addr_int <= to_integer(unsigned(lstm_cell_x_1_address)) + lstm_cell_x_1_address_offset;
    x_1_address <= std_logic_vector(to_unsigned(x_1_addr_int, x_1_address'length));
    x_2_address <= lstm_cell_x_2_address when read_states_from_prev_iter = false else (others=>'0');
    x_3_address <= lstm_cell_x_3_address when read_states_from_prev_iter = false else (others=>'0');
    lstm_cell_x_1_data <= x_1;
    lstm_cell_x_2_data <= x_2 when read_states_from_prev_iter=false else lstm_cell_y_1_data;
    lstm_cell_x_3_data <= x_3 when read_states_from_prev_iter=false else lstm_cell_y_2_data;
    y_2 <= lstm_cell_y_1_data;
    y_3 <= lstm_cell_y_2_data;
    fsm_process: process(clock, reset)
    begin
        if reset = '1' then
            loop_counter <= 0;
            lstm_cell_x_1_address_offset <= 0;
            cell_state <= s_stop;
            done <= '0';
            lstm_cell_enable <= '0';
        else
            if rising_edge(clock) then
                case cell_state is
                    when s_stop =>
                        cell_state <= s_start;
                    when s_start =>
                        lstm_cell_enable <= '1';
                        cell_state <= s_wait;
                    when s_wait =>
                        if lstm_cell_done = '1' then
                            cell_state <= s_read_out;
                        end if;
                    when s_read_out =>
                        if read_out_done = '1' then
                            if loop_counter = X_1_COUNT-1 then
                                cell_state <= s_done;
                            else
                                cell_state <= s_start;
                                loop_counter <= loop_counter + 1;
                                lstm_cell_x_1_address_offset <= lstm_cell_x_1_address_offset + NUM_DIMENSIONS;
                                lstm_cell_enable <= '0';
                            end if;
                        end if;
                    when s_done =>
                        loop_counter <= 0;
                        done <= '1';
                end case;
            end if;
        end if;
    end process;

    data_offload_process: process(clock, reset)
    variable read_out_counter : integer range 0 to Y_2_COUNT := 0;
    variable var_y_store_counter : integer range 0 to Y_1_COUNT := 0;
    variable delay : integer range 0 to 1 := 0;
    begin
        if rising_edge(clock) then
            if cell_state=s_stop then
                var_y_store_counter := 0;
                read_out_counter := 0;
            elsif cell_state = s_read_out then
                cell_y1_store_en <= '1';
                if delay = 0 then
                    if read_out_done='0' then
                        if read_out_counter < Y_2_COUNT-1 then
                            read_out_counter := read_out_counter + 1;
                            var_y_store_counter := var_y_store_counter + 1;
                            delay := 1;
                        else
                            read_out_done <= '1';
                            var_y_store_counter := var_y_store_counter + 1;
                        end if;
                    end if;
                else
                    delay := delay - 1;
                end if;
            else
                read_out_counter := 0;
                read_out_done <= '0';
                delay := 1;
                cell_y1_store_en <= '0';
            end if;

            cell_y1_read_addr <= std_logic_vector(to_unsigned(read_out_counter, cell_y1_read_addr'length));
            cell_y1_store_addr_std <= std_logic_vector(to_unsigned(var_y_store_counter, cell_y1_store_addr_std'length));
        end if;
    end process;

    temp_addr <= cell_y1_read_addr when lstm_cell_done='1' else lstm_cell_x_2_address;
    -- hidden states, from previous iteration
    lstm_cell_y_1_address <= y_2_address when cell_state=s_done else temp_addr;

    -- cell states, from previous iteration
    lstm_cell_y_2_address <= y_3_address when cell_state=s_done else lstm_cell_x_3_address; -- read_out

    y_2 <= lstm_cell_y_1_data;
    y_3 <= lstm_cell_y_2_data;

    inst_${cell_name}: entity ${work_library_name}.${cell_name}(rtl)
        port map (
            enable => lstm_cell_enable,
            clock  => lstm_cell_clock,
            x_1_address  => lstm_cell_x_1_address,
            x_2_address  => lstm_cell_x_2_address,
            x_3_address  => lstm_cell_x_3_address,
            x_1  => lstm_cell_x_1_data,
            x_2  => lstm_cell_x_2_data,
            x_3  => lstm_cell_x_3_data,
            y_1_address  => lstm_cell_y_1_address,
            y_2_address  => lstm_cell_y_2_address,
            y_1  => lstm_cell_y_1_data,
            y_2  => lstm_cell_y_2_data,
            done  => lstm_cell_done
        );

    cell_y1_store_data <= lstm_cell_y_1_data;
    ram_y1 : entity ${work_library_name}.${name}_ram(rtl)
    generic map (
        RAM_WIDTH => DATA_WIDTH,
        RAM_DEPTH_WIDTH => Y_1_ADDR_WIDTH,
        RAM_PERFORMANCE => "LOW_LATENCY",
        RESOURCE_OPTION => RESOURCE_OPTION,
        INIT_FILE => ""
    )
    port map  (
        addra  => cell_y1_store_addr_std,
        addrb  => y1_rd_out_addr,
        dina   => cell_y1_store_data,
        clka   => clock,
        clkb   => clock,
        wea    => cell_y1_store_en,
        enb    => '1',
        rstb   => '0',
        regceb => '1',
        doutb  => y1_rd_out_data
    );
    y_1 <= y1_rd_out_data; -- read_out
    y1_rd_out_addr <= y_1_address; -- read_out
end architecture;
