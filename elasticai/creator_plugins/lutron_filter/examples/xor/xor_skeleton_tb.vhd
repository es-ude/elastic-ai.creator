library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;
use work.counter_pkg.clog2;
library vunit_lib;
context vunit_lib.vunit_context;

entity xor_skeleton_tb is
    generic (runner_cfg: string);
end entity;

architecture behav of xor_skeleton_tb is
    signal clk : std_logic := '1';
    constant NUM_INPUT_BYTES : integer := 1;
    constant NUM_OUTPUT_BYTES : integer := 1;
    subtype byte_t is std_logic_vector(7 downto 0);
    type vector_t is array (natural range <>) of byte_t;
    type matrix_t is array (natural range <>) of vector_t;
    subtype y_t is vector_t (0 to NUM_OUTPUT_BYTES - 1);
    subtype x_t is vector_t (0 to NUM_INPUT_BYTES - 1);
    signal busy : std_logic := '1';
    signal rd : std_logic := '0';
    signal wr : std_logic := '0';
    signal data_out : std_logic_vector(7 downto 0);
    signal data_in : std_logic_vector(7 downto 0);
    signal address_in : std_logic_vector(15 downto 0);
    signal done : std_logic;
    signal reset : std_logic := '0';
    signal test_counter : natural := 0;
    constant clock_freq : time := 2 ps;
    constant clock_cycle: time := 2 * clock_freq;


    function increment_std_logic_vector(input: std_logic_vector) return std_logic_vector is
        variable number :integer;
    begin
        number := to_integer(unsigned(input));
        number := number + 1;
        return std_logic_vector(to_unsigned(number, input'length));
    end function;

    begin
        done  <= not busy;
        skeleton : entity work.skeleton(rtl)

            port map (
                data_in => data_in,
                data_out => data_out,
                address_in => address_in,
                rd => rd,
                wr => wr,
                clock => clk,
                clk_hadamard => clk,
                busy => busy,
                reset => reset
            );

        clk <= not clk after clock_freq;


        process  is
            procedure write_input_byte(constant byte: byte_t; constant num_byte: natural) is begin
                address_in <= std_logic_vector(to_unsigned(num_byte + 18, address_in'length));
                data_in <= byte;
            end procedure;

            procedure write_enable is begin
                wr <= '1';
                rd <= '0';
            end procedure;
            procedure read_enable is begin
                wr <= '0';
                rd <= '1';
            end procedure;
            procedure disable_read_write is begin
                wr <= '0';
                rd <= '0';
            end procedure;

            procedure disable_network is begin
                write_enable;
                address_in <= std_logic_vector(to_unsigned(16, address_in'length));
                data_in <= x"00";
                wait for clock_cycle;
            end procedure;

            procedure enable_network is begin
                write_enable;
                address_in <= std_logic_vector(to_unsigned(16, address_in'length));
                data_in <= x"ff";
                wait for clock_cycle;
            end procedure;

            procedure address_output(
                constant num_byte: natural
                )
            is begin
                address_in <= std_logic_vector(to_unsigned(18, address_in'length));
                wait for clock_cycle;
            end procedure;

            procedure reset_network is begin
                wait for clock_cycle;
                reset <= '1';
                wait for clock_cycle;
                reset <= '0';
                wait for clock_cycle;
            end procedure;

            procedure write_input(constant input: vector_t) is begin
                for byte_id in 0 to NUM_INPUT_BYTES - 1 loop
                    write_input_byte(input(byte_id), byte_id);
                    wait for clock_cycle;
                end loop;
            end procedure;

            procedure check_output_bytes(constant expected: vector_t) is begin
                for num_output in expected'left to expected'right loop
                    address_output(num_output);
                    wait for 2*clock_cycle;
                    check_equal(data_out, expected(num_output), "wrong inference result");
                end loop;
            end procedure;

            procedure count_cycles_until_done(variable counter: inout natural) is begin
              
                while done /= '1' loop
                    counter := counter + 1;
                    wait for clock_cycle;
                end loop;

            end procedure;

            procedure run_inference(
                constant input: vector_t;
                constant expected: vector_t
            ) is
                variable counter : integer := 0;
            begin

                disable_network;
                disable_read_write;
                reset_network;

                write_enable;


                write_input(input);

                enable_network;
                wait for clock_cycle;

                disable_read_write;

                count_cycles_until_done(counter);
                check_equal(counter, 1, "comparing clock cycles until done");
                read_enable;
                info("input bytes " & to_string(input(0)) &  " expected: " & to_string(expected(0)));
                check_output_bytes(expected);                

                write_enable;
                disable_network;
                test_counter <= test_counter + 1;
                wait for clock_cycle;

            end procedure;

            procedure run_inference(
                constant input : byte_t;
                constant expected : byte_t
            ) is
                constant input_v : vector_t := (0 => input);
                constant expected_v : vector_t := (0 => expected);
            begin
                run_inference(input_v, expected_v);
            end procedure;

        begin
            test_runner_setup(runner, runner_cfg);
            wait for 4*clock_cycle;

            run_inference(
                x"00",
                x"00"
            );
            run_inference(
                x"01",
                x"01"
            );
            run_inference(
                x"02",
                x"01"
            );
            run_inference(
                x"03",
                x"00"
            );
            run_inference(
                x"04",
                x"00"
            );
            run_inference(
                x"05",
                x"01"
            );
            run_inference(
                x"06",
                x"01"
            );
            run_inference(
                x"08",
                x"00"
            );
            run_inference(
                x"09",
                x"01"
            );
            test_runner_cleanup(runner);

            finish;
        end process;
end architecture;
