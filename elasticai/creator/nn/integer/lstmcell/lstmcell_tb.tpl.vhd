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
        X_1_ADDR_WIDTH : integer := ${x_1_addr_width};
        X_2_ADDR_WIDTH : integer := ${x_2_addr_width};
        X_3_ADDR_WIDTH : integer := ${x_3_addr_width};
        Y_1_ADDR_WIDTH : integer := ${y_1_addr_width};
        Y_2_ADDR_WIDTH : integer := ${y_2_addr_width};
        X_1_NUM_FEATURES : integer := ${x_1_num_features};
        X_1_NUM_DIMENSIONS : integer := ${x_1_num_dimensions};
        X_2_NUM_FEATURES : integer := ${x_2_num_features};
        X_2_NUM_DIMENSIONS : integer := ${x_2_num_dimensions};
        X_3_NUM_FEATURES : integer := ${x_3_num_features};
        X_3_NUM_DIMENSIONS : integer := ${x_3_num_dimensions};
        Y_1_NUM_FEATURES : integer := ${y_1_num_features};
        Y_1_NUM_DIMENSIONS : integer := ${y_1_num_dimensions};
        Y_2_NUM_FEATURES : integer := ${y_2_num_features};
        Y_2_NUM_DIMENSIONS : integer := ${y_2_num_dimensions}
    );
    port(
        clk : out std_logic
        );
end entity;
architecture rtl of ${name}_tb is
    constant C_CLK_PERIOD : time := 10 ns;
    signal clock : std_logic := '0';
    signal reset : std_logic := '0';
    signal uut_enable : std_logic := '0';
    signal x_1_address : std_logic_vector(X_1_ADDR_WIDTH - 1 downto 0);
    signal x_2_address : std_logic_vector(X_2_ADDR_WIDTH - 1 downto 0);
    signal x_3_address : std_logic_vector(X_3_ADDR_WIDTH - 1 downto 0);
    signal x_1 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal x_2 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal x_3 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    type t_array_x_1 is array (0 to 2**X_1_ADDR_WIDTH-1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
    type t_array_x_2 is array (0 to 2**X_2_ADDR_WIDTH-1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
    type t_array_x_3 is array (0 to 2**X_3_ADDR_WIDTH-1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal x_1_arr : t_array_x_1 := (others=>(others=>'0'));
    signal x_2_arr : t_array_x_2 := (others=>(others=>'0'));
    signal x_3_arr : t_array_x_3 := (others=>(others=>'0'));
    signal y_1_address : std_logic_vector(Y_1_ADDR_WIDTH - 1 downto 0);
    signal y_2_address : std_logic_vector(Y_2_ADDR_WIDTH - 1 downto 0);
    signal y_1 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal y_2 : std_logic_vector(DATA_WIDTH - 1 downto 0);
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
        constant file_inputs_x1:      string := "./data/stacked_rnnrnn_layer_0_lstm_cell_q_x_1.txt";
        constant file_inputs_x2:      string := "./data/stacked_rnnrnn_layer_0_lstm_cell_q_x_2.txt";
        constant file_inputs_x3:      string := "./data/stacked_rnnrnn_layer_0_lstm_cell_q_x_3.txt";
        constant file_labels_y1:      string := "./data/stacked_rnnrnn_layer_0_lstm_cell_q_y_1.txt";
        constant file_labels_y2:      string := "./data/stacked_rnnrnn_layer_0_lstm_cell_q_y_2.txt";
        constant file_pred_y1:      string := "./data/stacked_rnnrnn_layer_0_lstm_cell_q_out_1.txt";
        constant file_pred_y2:      string := "./data/stacked_rnnrnn_layer_0_lstm_cell_q_out_2.txt";
        file fp_inputs_x1:      text;
        file fp_inputs_x2:      text;
        file fp_inputs_x3:      text;
        file fp_labels_y1:      text;
        file fp_labels_y2:      text;
        file fp_pred_y1:      text;
        file fp_pred_y2:      text;
        variable line_content:  integer;
        variable line_num:      line;
        variable filestatus:    file_open_status;
        variable input_rd_cnt : integer := 0;
        variable output_rd_cnt : integer := 0;
        variable v_TIME : time := 0 ns;
    begin
        file_open (filestatus, fp_inputs_x1, file_inputs_x1, READ_MODE);
        report file_inputs_x1 & LF & HT & "file_open_status = " &
                    file_open_status'image(filestatus);
        assert filestatus = OPEN_OK
            report "file_open_status /= file_ok"
            severity FAILURE;
        file_open (filestatus, fp_inputs_x2, file_inputs_x2, READ_MODE);
        report file_inputs_x2 & LF & HT & "file_open_status = " &
                    file_open_status'image(filestatus);
        assert filestatus = OPEN_OK
            report "file_open_status /= file_ok"
            severity FAILURE;
        file_open (filestatus, fp_inputs_x3, file_inputs_x3, READ_MODE);
        report file_inputs_x3 & LF & HT & "file_open_status = " &
                    file_open_status'image(filestatus);
        assert filestatus = OPEN_OK
            report "file_open_status /= file_ok"
            severity FAILURE;

        file_open (filestatus, fp_labels_y1, file_labels_y1, READ_MODE);
        report file_labels_y1 & LF & HT & "file_open_status = " &
                    file_open_status'image(filestatus);
        assert filestatus = OPEN_OK
            report "file_open_status /= file_ok"
            severity FAILURE;
        file_open (filestatus, fp_labels_y2, file_labels_y2, READ_MODE);
        report file_labels_y2 & LF & HT & "file_open_status = " &
                    file_open_status'image(filestatus);
        assert filestatus = OPEN_OK
            report "file_open_status /= file_ok"
            severity FAILURE;

        file_open (filestatus, fp_pred_y1, file_pred_y1, WRITE_MODE);
        report file_pred_y1 & LF & HT & "file_open_status = " &
                    file_open_status'image(filestatus);
        assert filestatus = OPEN_OK
            report "file_open_status /= file_ok"
            severity FAILURE;
        file_open (filestatus, fp_pred_y2, file_pred_y2, WRITE_MODE);
        report file_pred_y2 & LF & HT & "file_open_status = " &
                    file_open_status'image(filestatus);
        assert filestatus = OPEN_OK
            report "file_open_status /= file_ok"
            severity FAILURE;

        y_1_address <= (others=>'0');
        y_2_address <= (others=>'0');
        uut_enable <= '0';
        wait until reset='0';
        wait for C_CLK_PERIOD;
        while not ENDFILE (fp_inputs_x1) loop
            input_rd_cnt := 0;
            while input_rd_cnt < X_1_NUM_FEATURES * X_1_NUM_DIMENSIONS loop
                readline (fp_inputs_x1, line_num);
                read (line_num, line_content);
                x_1_arr(input_rd_cnt) <= std_logic_vector(to_signed(line_content, DATA_WIDTH));
                input_rd_cnt := input_rd_cnt + 1;
            end loop;
            while input_rd_cnt < X_2_NUM_FEATURES * X_2_NUM_DIMENSIONS loop
                readline (fp_inputs_x2, line_num);
                read (line_num, line_content);
                x_2_arr(input_rd_cnt) <= std_logic_vector(to_signed(line_content, DATA_WIDTH));
                input_rd_cnt := input_rd_cnt + 1;
            end loop;
            while input_rd_cnt < X_3_NUM_FEATURES * X_3_NUM_DIMENSIONS loop
                readline (fp_inputs_x3, line_num);
                read (line_num, line_content);
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
            while output_rd_cnt< Y_1_NUM_FEATURES * Y_1_NUM_DIMENSIONS loop
                readline (fp_labels_y1, line_num);
                read (line_num, line_content);
                y_1_address <= std_logic_vector(to_unsigned(output_rd_cnt, y_1_address'length));
                wait for 2*C_CLK_PERIOD;
                report "Correct/Simulated = " & integer'image(line_content) & "/" & integer'image(to_integer(signed(y_1))) & ", Differece = " & integer'image(line_content - to_integer(signed(y_1)));
                write (line_num, to_integer(signed(y_1)));
                writeline(fp_pred_y1, line_num);
                output_rd_cnt := output_rd_cnt + 1;
            end loop;
            while output_rd_cnt< Y_2_NUM_FEATURES * Y_2_NUM_DIMENSIONS loop
                readline (fp_labels_y2, line_num);
                read (line_num, line_content);
                y_2_address <= std_logic_vector(to_unsigned(output_rd_cnt, y_2_address'length));
                wait for 2*C_CLK_PERIOD;
                report "Correct/Simulated = " & integer'image(line_content) & "/" & integer'image(to_integer(signed(y_2))) & ", Differece = " & integer'image(line_content - to_integer(signed(y_2)));
                write (line_num, to_integer(signed(y_2)));
                writeline(fp_pred_y2, line_num);
                output_rd_cnt := output_rd_cnt + 1;
            end loop;
            uut_enable <= '0';
        end loop;
        wait until falling_edge(clock);
        file_close (fp_inputs_x1);
        file_close (fp_inputs_x2);
        file_close (fp_inputs_x3);
        file_close (fp_labels_y1);
        file_close (fp_labels_y2);
        file_close (fp_pred_y1);
        file_close (fp_pred_y2);
        report "All files closed.";
        report "Time taken for processing = " & time'image(v_TIME);
        report "Simulation done.";
        assert false report "Simulation done. The `assertion failure` is intended to stop this simulation." severity FAILURE;
    end process ;
    uut: entity work.${name}(rtl)
    port map (
        enable => uut_enable,
        clock  => clock,
        x_1_address => x_1_address,
        x_2_address => x_2_address,
        x_3_address => x_3_address,
        y_1_address => y_1_address,
        y_2_address => y_2_address,
        x_1   => x_1,
        x_2   => x_2,
        x_3   => x_3,
        y_1  => y_1,
        y_2  => y_2,
        done   => done
    );
end architecture;
