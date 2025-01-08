library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name}_tb is
    generic (
        X_ADDR_WIDTH : integer := ${x_address_width};
        Y_ADDR_WIDTH : integer := ${y_address_width};
        DATA_WIDTH : integer := ${data_width};
        X_COUNT : integer := ${x_count};
        Y_COUNT : integer := ${y_count}
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
    signal x_addr : std_logic_vector(X_ADDR_WIDTH - 1 downto 0);
    signal x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal y_addr : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
    signal y_out : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal done : std_logic;
    type t_array_x is array (0 to X_COUNT - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal x_arr : t_array_x := (others=>(others=>'0'));
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
            x_in <= x_arr(to_integer(unsigned(x_addr)));
        end if;
    end process ;
    test_main : process
        constant file_inputs:      string := "./data/${name}_q_x.txt";
        constant file_labels:      string := "./data/${name}_q_y.txt";
        constant file_pred:      string := "./data/${name}_out.txt";
        file fp_inputs:      text;
        file fp_labels:      text;
        file fp_pred:      text;
        variable line_content:  integer;
        variable line_num:      line;
        variable filestatus:    file_open_status;
        variable input_rd_cnt : integer := 0;
        variable output_rd_cnt : integer := 0;
        variable v_TIME : time := 0 ns;
    begin
        file_open (filestatus, fp_inputs, file_inputs, READ_MODE);
        report file_inputs & LF & HT & "file_open_status = " &
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
        y_addr <= (others=>'0');
        uut_enable <= '0';
        wait until reset='0';
        wait for C_CLK_PERIOD;
        while not ENDFILE (fp_inputs) loop
            input_rd_cnt := 0;
            while input_rd_cnt < X_COUNT loop
                readline (fp_inputs, line_num);
                read (line_num, line_content);
                x_arr(input_rd_cnt) <= std_logic_vector(to_signed(line_content, DATA_WIDTH));
                input_rd_cnt := input_rd_cnt + 1;
            end loop;
            wait for C_CLK_PERIOD;
            v_TIME := now;
            uut_enable <= '1';
            wait for C_CLK_PERIOD;
            wait until done='1';
            v_TIME := now - v_TIME;
            output_rd_cnt := 0;
            while output_rd_cnt<Y_COUNT loop
                readline (fp_labels, line_num);
                read (line_num, line_content);
                y_addr <= std_logic_vector(to_unsigned(output_rd_cnt, y_addr'length));
                wait for 2*C_CLK_PERIOD;
                report "Correct/Simulated = " & integer'image(line_content) & "/" & integer'image(to_integer(signed(y_out))) & ", Differece = " & integer'image(line_content - to_integer(signed(y_out)));
                write (line_num, to_integer(signed(y_out)));
                writeline(fp_pred, line_num);
                output_rd_cnt := output_rd_cnt + 1;
            end loop;
            uut_enable <= '0';
        end loop;
        wait until falling_edge(clock);
        file_close (fp_inputs);
        file_close (fp_labels);
        file_close (fp_pred);
        report  "all files closed.";
        report "Time taken for processing = " & time'image(v_TIME);
        report "Simulation done.";
        assert false report "Simulation done. The `assertion failure` is intended to stop this simulation." severity FAILURE;
    end process ;
    uut: entity ${work_library_name}.${name}(rtl)
    port map (
        enable => uut_enable,
        clock  => clock,
        x_address => x_addr,
        y_address => y_addr,
        x   => x_in,
        y   => y_out,
        done   => done
    );
end architecture;
