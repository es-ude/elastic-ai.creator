library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;
use std.env.finish;


entity $name is
  generic (
      INPUTS_FILE_PATH: string
  );
end;

architecture Behavioral of $name is
    signal clock_period : time := 2 ps;
    signal clock : std_logic := '0';
    signal reset : std_logic := '0';
    signal next_sample : std_logic;
    signal x1 : std_logic_vector(0 downto 0) := (others => '0');
    signal x2 : std_logic_vector(0 downto 0) := (others => '0');
    signal sum :  std_logic;
    signal done :  std_logic;

    type data is array (0 to 100) of std_logic_vector(0 downto 0);
    signal data_in_x1 : data;
    signal data_in_x2 : data;
    file input_file : text open read_mode is INPUTS_FILE_PATH;


begin
    UUT : entity work.${uut_name}
    port map (reset => reset, next_sample => next_sample, x1 => x1, x2 => x2, sum => sum, done => done);

    clk : process
    begin
        clock <= not clock;
        wait for clock_period/2;
    end process;


    testbench_1 : process(clock, done)

        variable v_ILINE     : line;
        variable v_in : std_logic_vector(0 downto 0);
        variable v_SPACE     : character;
        variable input_idx : integer range 0 to 100 := 0;
        variable num_inputs : integer range 0 to 100:= 0;
        type TYPE_STATE is (s_start_up, s_load_batch, s_reset, s_start_computation, s_wait_for_computation_done, s_read_uut_output, s_finish_simulation);
        variable test_state : TYPE_STATE := s_start_up;

    begin
        if rising_edge(clock) then
            if test_state = s_start_up then
                report "status: reading file " & INPUTS_FILE_PATH;
                readline (input_file, v_ILINE); --read header
                test_state := s_load_batch;
            elsif test_state = s_load_batch then
                if input_idx = 0 then
                    report "status: start reading batch";
                    readline(input_file, v_ILINE);
                end if;
                if v_ILINE'length >= 3 then
                    read(v_ILINE, v_in); -- read value
                    report "debug: reading x1 " & to_bstring(v_in);
                    data_in_x1(input_idx) <= v_in;
                    read(v_ILINE, v_SPACE);

                    read(v_ILINE, v_in); -- read value
                    report "debug: reading x2 " & to_bstring(v_in);
                    data_in_x2(input_idx) <= v_in;
                    if v_ILINE'length > 0 then
                        read(v_ILINE, v_SPACE);
                    end if;
                else
                    report "status: data for batch loaded";
                    test_state := s_reset;
                    num_inputs := input_idx;
                end if;
                input_idx := input_idx + 1;
            elsif test_state = s_reset then
                input_idx := 0;
                reset <= '0';
                x1 <= (others => '0');
                x2 <= (others => '0');
                test_state := s_start_computation;
                next_sample <= '0';
            elsif test_state = s_start_computation then
                reset <= '1';
                x1 <= data_in_x1(input_idx);
                x2 <= data_in_x2(input_idx);
                next_sample <= '1';
                test_state := s_wait_for_computation_done;
            elsif test_state = s_wait_for_computation_done then
                next_sample <= '0';
                if done = '1' then
                    if input_idx = num_inputs then
                        input_idx := 0;
                        test_state := s_read_uut_output;
                    else
                        input_idx := input_idx +1;
                        test_state := s_start_computation;
                    end if;
                end if;
            elsif test_state = s_read_uut_output then
                report "result: " & to_string(sum);
                if endfile(input_file) then
                    test_state := s_finish_simulation;
                else
                    test_state := s_load_batch;
                end if;
            elsif test_state = s_finish_simulation then
                report "status: simulation finished";
                finish;
            end if;
        end if;
    end process;

    end_after_X_cycles : process (clock)
    variable i : integer range 0 to 10000;
    begin
        if rising_edge(clock) then
            if i = 10000 then
                report("OUT of TIME");
                finish;
            else
                i := i + 1;
            end if;
        end if;
    end process;

end Behavioral;
