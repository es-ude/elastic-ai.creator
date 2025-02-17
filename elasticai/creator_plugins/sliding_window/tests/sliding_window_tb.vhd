library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;
library vunit_lib;
context vunit_lib.vunit_context;

entity sliding_window_tb is
    generic (runner_cfg : string);
end entity;

architecture behav of sliding_window_tb is
    constant KERNEL_SIZE : natural := 3;
    constant NUM_CHANNELS : natural := 3;
    constant NUM_STEPS : natural := 1;
    signal rst : std_logic := '0';
    constant TOTAL_LENGTH : natural := (KERNEL_SIZE + NUM_STEPS - 1 ) * NUM_CHANNELS;
    signal d_in : std_logic_vector(12 - 1 downto 0) := (others => 'X'); 
    signal d_out : std_logic_vector(3 - 1 downto 0) := (others => 'X');
    signal valid_in : std_logic := '0';
    constant clk_period : time := 10 ns;
    signal clk : std_logic := '1';
    signal valid_in_stride : std_logic := '0';
    signal d_out_stride : std_logic_vector(3 - 1 downto 0) := (others => 'X');
    begin

      
        dut_i : entity work.sliding_window(rtl)
            generic map (
                INPUT_WIDTH => 12,
                OUTPUT_WIDTH => 3
            )
            port map (
                d_in => d_in,
                d_out => d_out,
                clk => clk,
                valid_in => valid_in,
                rst => rst
            );

        sliding_window_with_stride: entity work.sliding_window(rtl)
            generic map (
                INPUT_WIDTH => 12,
                OUTPUT_WIDTH => 3,
                STRIDE => 3
            )
            port map (
                d_in => d_in,
                d_out => d_out_stride,
                clk => clk,
                valid_in => valid_in_stride,
                rst => rst
            );
        clk <= not clk after clk_period / 2;
        

        testing: process is
            type expected_t is array (natural range <>) of std_logic_vector(3 - 1 downto 0);

            procedure reset_window is begin
                rst <= '1';
                wait for clk_period;
                rst <= '0';
                wait for clk_period;
            end procedure;

            procedure check_output_sequence(
                input: in std_logic_vector(d_in'range);
                expected : expected_t;
                signal output: std_logic_vector
            ) is
            begin
                d_in <= input;
                wait until rising_edge(clk);
                for i in expected'range loop
                    check_equal(output, expected(i), "at counter=" & to_string(i));
                    wait for clk_period;
                end loop;
            end procedure;
        begin
            test_runner_setup(runner, runner_cfg);
            if run("check output sequences w/o stride") then
                valid_in <= '0';
                reset_window;
                valid_in <= '1';
                check_output_sequence(b"111011101010",
                (b"111", b"110", b"101", b"011"), d_out);
                valid_in <= '0';
                reset_window;
                valid_in <= '1';
                check_output_sequence(
                    b"001000100000",
                    (b"001", b"010", b"100", b"000",
                    b"001", b"010", b"100", b"000"),
                    d_out
                );
                valid_in <= '0';


            end if;

            if run("check output sequence with stride 3") then
                reset_window;
                valid_in_stride <= '1';
                check_output_sequence(b"111011101010",(b"111", b"011",
                b"101", b"010"), d_out_stride);
                valid_in_stride <= '0';
                reset_window;
                valid_in_stride <= '1';
                check_output_sequence(
                    b"001000100000",
                    (b"001", b"000", b"100", b"000"),
                    d_out_stride
                );
                valid_in_stride <= '0';
            end if;
            test_runner_cleanup(runner);

        end process;
end architecture;
