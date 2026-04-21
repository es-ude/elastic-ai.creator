library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name}_tb is
    generic (
        X_1_ADDR_WIDTH : integer := ${x_1_addr_width};
        X_2_ADDR_WIDTH : integer := ${x_2_addr_width};
        X_3_ADDR_WIDTH : integer := ${x_3_addr_width};
        Y_ADDR_WIDTH : integer := ${y_addr_width};
        DATA_WIDTH : integer := ${data_width};
        MATMUL_SCORE_X_1_DIM_A : integer := ${matmul_score_x_1_dim_a};
        MATMUL_SCORE_X_1_DIM_B : integer := ${matmul_score_x_1_dim_b};
        MATMUL_SCORE_X_1_DIM_C : integer := ${matmul_score_x_1_dim_c};
        MATMUL_SCORE_X_2_DIM_A : integer := ${matmul_score_x_2_dim_a};
        MATMUL_SCORE_X_2_DIM_B : integer := ${matmul_score_x_2_dim_b};
        MATMUL_SCORE_X_2_DIM_C : integer := ${matmul_score_x_2_dim_c};
        MATMUL_ATT_X_2_DIM_A : integer := ${matmul_att_x_2_dim_a};
        MATMUL_ATT_X_2_DIM_B : integer := ${matmul_att_x_2_dim_b};
        MATMUL_ATT_X_2_DIM_C : integer := ${matmul_att_x_2_dim_c};
        MATMUL_ATT_Y_DIM_A : integer := ${matmul_att_y_dim_a};
        MATMUL_ATT_Y_DIM_B : integer := ${matmul_att_y_dim_b};
        MATMUL_ATT_Y_DIM_C : integer := ${matmul_att_y_dim_c}
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
    signal x_1_address : std_logic_vector(X_1_ADDR_WIDTH-1 downto 0) := (others => '0');
    signal x_1 : std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
    type t_array_x_1 is array (0 to MATMUL_SCORE_X_1_DIM_A * MATMUL_SCORE_X_1_DIM_B * MATMUL_SCORE_X_1_DIM_C - 1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal x_1_arr : t_array_x_1 := (others => (others => '0'));
    signal x_2_address : std_logic_vector(X_2_ADDR_WIDTH-1 downto 0) := (others => '0');
    signal x_2 : std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
    type t_array_x_2 is array (0 to MATMUL_SCORE_X_2_DIM_A * MATMUL_SCORE_X_2_DIM_B * MATMUL_SCORE_X_2_DIM_C - 1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal x_2_arr : t_array_x_2 := (others => (others => '0'));
    signal x_3_address : std_logic_vector(X_3_ADDR_WIDTH-1 downto 0) := (others => '0');
    signal x_3 : std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
    type t_array_x_3 is array (0 to MATMUL_ATT_X_2_DIM_A * MATMUL_ATT_X_2_DIM_B * MATMUL_ATT_X_2_DIM_C - 1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal x_3_arr : t_array_x_3 := (others => (others => '0'));
    signal y_address : std_logic_vector(Y_ADDR_WIDTH-1 downto 0) := (others => '0');
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
            while input_rd_cnt < MATMUL_SCORE_X_1_DIM_A * MATMUL_SCORE_X_1_DIM_B * MATMUL_SCORE_X_1_DIM_C loop
                readline(fp_inputs_1, line_num);
                read(line_num, line_content);
                x_1_arr(input_rd_cnt) <= std_logic_vector(to_signed(line_content, DATA_WIDTH));
                input_rd_cnt := input_rd_cnt + 1;
            end loop;
            input_rd_cnt := 0;
            while input_rd_cnt < MATMUL_SCORE_X_2_DIM_A * MATMUL_SCORE_X_2_DIM_B * MATMUL_SCORE_X_2_DIM_C loop
                readline(fp_inputs_2, line_num);
                read(line_num, line_content);
                x_2_arr(input_rd_cnt) <= std_logic_vector(to_signed(line_content, DATA_WIDTH));
                input_rd_cnt := input_rd_cnt + 1;
            end loop;
            input_rd_cnt := 0;
            while input_rd_cnt < MATMUL_ATT_X_2_DIM_A * MATMUL_ATT_X_2_DIM_B * MATMUL_ATT_X_2_DIM_C loop
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
            while output_rd_cnt < MATMUL_ATT_Y_DIM_A * MATMUL_ATT_Y_DIM_B * MATMUL_ATT_Y_DIM_C loop
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
    uut:  entity ${work_library_name}.${name}(rtl)
    port map (
        enable => uut_enable,
        clock => clock,
        x_1_address => x_1_address,
        x_1 => x_1,
        x_2_address => x_2_address,
        x_2 => x_2,
        x_3_address => x_3_address,
        x_3 => x_3,
        y_address => y_address,
        y => y,
        done => done
    );
end architecture;
