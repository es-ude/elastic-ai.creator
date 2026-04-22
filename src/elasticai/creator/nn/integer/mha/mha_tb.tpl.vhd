library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name}_tb is
    generic (
        DATA_WIDTH : integer := ${data_width};
        Q_LINEAR_X_ADDR_WIDTH : integer := ${q_linear_x_addr_width};
        Q_LINEAR_NUM_DIMENSIONS : integer := ${q_linear_num_dimensions};
        Q_LINEAR_IN_FEATURES : integer := ${q_linear_in_features};
        K_LINEAR_X_ADDR_WIDTH : integer := ${k_linear_x_addr_width};
        K_LINEAR_NUM_DIMENSIONS : integer := ${k_linear_num_dimensions};
        K_LINEAR_IN_FEATURES : integer := ${k_linear_in_features};
        V_LINEAR_X_ADDR_WIDTH : integer := ${v_linear_x_addr_width};
        V_LINEAR_NUM_DIMENSIONS : integer := ${v_linear_num_dimensions};
        V_LINEAR_IN_FEATURES : integer := ${v_linear_in_features};
        OUTPUT_LINEAR_Y_ADDR_WIDTH : integer := ${output_linear_y_addr_width};
        OUTPUT_LINEAR_NUM_DIMENSIONS : integer := ${output_linear_num_dimensions};
        OUTPUT_LINEAR_OUT_FEATURES : integer := ${output_linear_out_features}
    );
    port (
        clk : out std_logic
    );
end entity;
architecture rtl of ${name}_tb is
    constant C_CLK_PERIOD : time := 10 ns;
    signal clock : std_logic := '0';
    signal reset : std_logic := '0';
    signal uut_enable : std_logic := '0';
    signal x_1_address : std_logic_vector(Q_LINEAR_X_ADDR_WIDTH-1 downto 0) := (others => '0');
    signal x_1 : std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
    type t_array_x_1 is array (0 to Q_LINEAR_NUM_DIMENSIONS * Q_LINEAR_IN_FEATURES -1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal x_1_arr : t_array_x_1 := (others => (others => '0'));
    signal x_2_address : std_logic_vector(K_LINEAR_X_ADDR_WIDTH-1 downto 0) := (others => '0');
    signal x_2 : std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
    type t_array_x_2 is array (0 to K_LINEAR_NUM_DIMENSIONS * K_LINEAR_IN_FEATURES -1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal x_2_arr : t_array_x_2 := (others => (others => '0'));
    signal x_3_address : std_logic_vector(V_LINEAR_X_ADDR_WIDTH-1 downto 0) := (others => '0');
    signal x_3 : std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
    type t_array_x_3 is array (0 to V_LINEAR_NUM_DIMENSIONS * V_LINEAR_IN_FEATURES -1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal x_3_arr : t_array_x_3 := (others => (others => '0'));
    signal y_address : std_logic_vector(OUTPUT_LINEAR_Y_ADDR_WIDTH-1 downto 0) := (others => '0');
    signal y : std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
    signal done : std_logic;
    begin
        CLK_GEN : process
        begin
            clock <= '1';
            wait for C_CLK_PERIOD/2;
            clock <= '0';
            wait for C_CLK_PERIOD/2;
        end process CLK_GEN;
        RESET_GEN : process
        begin
            reset <= '1',
                    '0' after 20.0*C_CLK_PERIOD;
        wait;
        end process RESET_GEN;

        clk <= clock;
        data_read : process( clock )
        begin
            if rising_edge(clock) then
                x_1 <= x_1_arr(to_integer(unsigned(x_1_address)));
                x_2 <= x_2_arr(to_integer(unsigned(x_2_address)));
                x_3 <= x_3_arr(to_integer(unsigned(x_3_address)));
            end if;
        end process ;
        test_main : process
        constant file_inputs_1 :    string := "./data/${name}_q_q.txt";
        constant file_inputs_2 :    string := "./data/${name}_q_k.txt";
        constant file_inputs_3 :    string := "./data/${name}_q_v.txt";
        constant file_labels :      string := "./data/${name}_q_y.txt";
        constant file_pred :        string := "./data/${name}_q_out.txt";
        file fp_inputs_1 :           text;
        file fp_inputs_2 :           text;
        file fp_inputs_3 :           text;
        file fp_labels :             text;
        file fp_pred :               text;
        variable line_content:  integer;
        variable line_num:      line;
        variable filestatus:    file_open_status;
        variable input_rd_cnt : integer := 0;
        variable output_rd_cnt : integer := 0;
        variable v_TIME : time := 0 ns;
    begin
        file_open (filestatus, fp_inputs_1, file_inputs_1, READ_MODE);
        report file_inputs_1 & LF & HT & "file_open_status = " &
                    file_open_status'image(filestatus);
        assert filestatus = OPEN_OK
            report "file_open_status /= file_ok"
            severity FAILURE;
        file_open (filestatus, fp_inputs_2, file_inputs_2, READ_MODE);
        report file_inputs_2 & LF & HT & "file_open_status = " &
                    file_open_status'image(filestatus);
        assert filestatus = OPEN_OK
            report "file_open_status /= file_ok"
            severity FAILURE;
        file_open (filestatus, fp_inputs_3, file_inputs_3, READ_MODE);
        report file_inputs_3 & LF & HT & "file_open_status = " &
                    file_open_status'image(filestatus);
        assert filestatus = OPEN_OK
            report "file_open_status /= file_ok"
            severity FAILURE;
        file_open (filestatus, fp_labels, file_labels, READ_MODE);
            report file_labels & LF & HT & "file_open_status = " &
                        file_open_status'image(filestatus);
            assert filestatus = OPEN_OK
                report "file_open_status /= file_ok"
                severity FAILURE;
        file_open (filestatus, fp_pred, file_pred, WRITE_MODE);
            report file_pred & LF & HT & "file_open_status = " &
                        file_open_status'image(filestatus);
            assert filestatus = OPEN_OK
                report "file_open_status /= file_ok"
                severity FAILURE;
        y_address <= (others => '0');
        uut_enable <= '0';
        wait until reset='0';
        wait for C_CLK_PERIOD;
        while not ENDFILE (fp_inputs_1) loop
            input_rd_cnt := 0;
            while input_rd_cnt < Q_LINEAR_NUM_DIMENSIONS * Q_LINEAR_IN_FEATURES loop
                readline(fp_inputs_1, line_num);
                read(line_num, line_content);
                x_1_arr(input_rd_cnt) <= std_logic_vector(to_signed(line_content, DATA_WIDTH));
                input_rd_cnt := input_rd_cnt + 1;
            end loop;
            input_rd_cnt := 0;
            while input_rd_cnt < K_LINEAR_NUM_DIMENSIONS * K_LINEAR_IN_FEATURES loop
                readline(fp_inputs_2, line_num);
                read(line_num, line_content);
                x_2_arr(input_rd_cnt) <= std_logic_vector(to_signed(line_content, DATA_WIDTH));
                input_rd_cnt := input_rd_cnt + 1;
            end loop;
            input_rd_cnt := 0;
            while input_rd_cnt < V_LINEAR_NUM_DIMENSIONS * V_LINEAR_IN_FEATURES loop
                readline(fp_inputs_3, line_num);
                read(line_num, line_content);
                x_3_arr(input_rd_cnt) <= std_logic_vector(to_signed(line_content, DATA_WIDTH));
                input_rd_cnt := input_rd_cnt + 1;
            end loop;
            wait for C_CLK_PERIOD;
            v_TIME := now;
            uut_enable <= '1';
            wait for C_CLK_PERIOD;
            wait until done='1';
            v_TIME := now - v_TIME;
            output_rd_cnt := 0;
            while output_rd_cnt < OUTPUT_LINEAR_NUM_DIMENSIONS * OUTPUT_LINEAR_OUT_FEATURES loop
                readline(fp_labels, line_num);
                read(line_num, line_content);
                y_address <= std_logic_vector(to_unsigned(output_rd_cnt, y_address'length));
                wait for 3*C_CLK_PERIOD;
                report "Correct/Simulated = " & integer'image(line_content) & "/" & integer'image(to_integer(signed(y))) & ", Differece = " & integer'image(line_content - to_integer(signed(y)));
                write (line_num, to_integer(signed(y)));
                writeline(fp_pred, line_num);
                output_rd_cnt := output_rd_cnt + 1;
            end loop;
            uut_enable <= '0';
        end loop;
        wait until falling_edge(clock);
        file_close(fp_inputs_1);
        file_close(fp_inputs_2);
        file_close(fp_inputs_3);
        file_close(fp_labels);
        file_close(fp_pred);
        report "All files closed.";
        report "Time taken for processing = " & time'image(v_TIME);
        report "Simulation done.";
        assert false report "Simulation done. The `assertion failure` is intended to stop this simulation." severity FAILURE;
    end process ;
    uut: entity ${work_library_name}.${name}(rtl)
    port map (
        enable => uut_enable,
        clock => clock,
        x_1_address => x_1_address,
        x_2_address => x_2_address,
        x_3_address => x_3_address,
        y_address => y_address,
        x_1 => x_1,
        x_2 => x_2,
        x_3 => x_3,
        y => y,
        done => done
    );
end architecture;
