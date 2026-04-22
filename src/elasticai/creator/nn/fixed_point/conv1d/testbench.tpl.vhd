library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;
use std.env.finish;

entity ${testbench_name} is
  generic (
      INPUTS_FILE_PATH: string
  );
end;

architecture rtl of ${testbench_name} is

    --CLOCK
    signal clock_period : time := 2 ns;

    --DATA INPUT
    type data is array (0 to ${input_signal_length}*${in_channels}-1) of std_logic_vector(${total_bits}-1 downto 0);
    signal data_in : data;
    file input_file : text open read_mode is INPUTS_FILE_PATH;

    --UUT
    signal clock : std_logic := '0';
    signal enable : std_logic := '0';
    signal x : std_logic_vector(${total_bits}-1 downto 0);
    signal x_address : std_logic_vector(${x_address_width}-1 downto 0) := (others => '0');
    signal x_address_std : std_logic_vector(${x_address_width}-1 downto 0);
    signal y : std_logic_vector(${total_bits}-1 downto 0);
    signal y_address : std_logic_vector(${y_address_width}-1 downto 0) := (others => '0');
    signal y_address_std : std_logic_vector(${y_address_width}-1 downto 0);
    signal done : std_logic;

begin
    x_address <= x_address_std;
    y_address_std <= y_address;

    UUT : entity work.${uut_name}
    port map (clock => clock, enable => enable, x => x, x_address => x_address_std, y => y, y_address => y_address_std, done => done);

    x_writing : process (clock)
    begin
        if falling_edge(clock) then
            -- After the layer in at idle mode, x is readable
            -- but it only update at the falling edge of the clock
            --report("debug: testbench: x_address "  & to_bstring(x_address));
            x <= data_in(to_integer(unsigned(x_address)));
        end if;
    end process x_writing;



    clk : process
    begin
        clock <= not clock;
        wait for clock_period/2;
    end process;

    start_test : process (clock)
        variable v_ILINE     : line;
        variable v_in : std_logic_vector(${total_bits}-1 downto 0);
        variable v_SPACE     : character;
        variable input_idx : integer range 0 to ${input_signal_length}*${in_channels} := 0;
        type TYPE_STATE is (s_start_up, s_load_batch, s_reset, s_start_computation, s_wait_for_computation_done, s_write_uut_output_address, s_read_uut_output, s_finish_simulation);
        variable test_state : TYPE_STATE := s_start_up;
        variable input_cycles : signed(7 downto 0) := (others => '0'); --max for 255 lines of inputs
    begin
        if rising_edge(clock) then
            if test_state = s_start_up then
                report "status: reading file " & INPUTS_FILE_PATH;
                --file_open(input_file, INPUTS_FILE_PATH, read_mode);
                readline(input_file, v_ILINE); -- read header
                test_state := s_load_batch;
            elsif test_state = s_load_batch then
                if endfile(input_file) and input_idx = ${input_signal_length}*${in_channels} then
                    test_state := s_finish_simulation;
                else
                    if input_idx = 0 then
                        report "status: start reading batch";
                        readline(input_file, v_ILINE);
                    end if;
                    read(v_ILINE, v_in); -- read value
                    data_in(input_idx) <= v_in;
                    report("status: reading " & to_bstring(v_in));
                    if input_idx /= ${input_signal_length}*${in_channels}-1 then
                        read(v_ILINE, v_SPACE);
                    else
                        report "status: data for batch loaded!";
                        test_state := s_reset;
                    end if;
                    input_idx := input_idx + 1;
                end if;
            elsif test_state = s_reset then
                report "status: test_state = s_reset";
                enable <= '0';
                test_state := s_start_computation;
                y_address <= (others => '0');
            elsif test_state = s_start_computation then
                report "status: test_state = s_start_computation";
                enable <= '1';
                test_state := s_wait_for_computation_done;
            elsif test_state = s_wait_for_computation_done then
                --report "status: test_state = s_wait_for_computation_done";
                if done = '1' then
                    report "status: computation finished";
                    test_state := s_read_uut_output;
                    y_address <= (others => '0');
                    enable <= '0';
                end if;
            elsif test_state = s_write_uut_output_address then
                report("status: test_state = s_write_uut_output_address");
                y_address <= std_logic_vector(unsigned(y_address) + 1);
                test_state := s_read_uut_output;
            elsif test_state = s_read_uut_output then
                report("status: test_state = s_read_uut_output");
                report("status: " & to_bstring(input_cycles) & "," & to_bstring(y));
                report("result: " & to_bstring(input_cycles) & "," & to_bstring(y));
                if unsigned(y_address) /= ${output_signal_length}*${out_channels}-1 then
                    test_state := s_write_uut_output_address;
                else
                    input_cycles := input_cycles + 1;
                    test_state := s_load_batch;
                end if;
            elsif test_state = s_finish_simulation then
                report "status: test_state = s_finish_simulation";
                report "status: simulation finished";
                finish;
            end if;
        end if;
    end process;

    end_after_100cycles : process (clock)
    variable i : integer range 0 to 10000;
    begin
        if rising_edge(clock) then
            if i = 200 then
                report("OUT of TIME");
                finish;
            else
                i := i + 1;
            end if;
        end if;
    end process;


end;
