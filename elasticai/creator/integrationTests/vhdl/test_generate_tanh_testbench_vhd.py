from functools import partial

from elasticai.creator.tests.vhdl.vhdl_file_testcase import GeneratedVHDLCodeTest
from elasticai.creator.vhdl.number_representations import float_values_to_fixed_point
from elasticai.creator.vhdl.precomputed_scalar_function import (
    PrecomputedScalarTestBench,
)


class TanhTestBenchTest(GeneratedVHDLCodeTest):
    def test_compare_files(self) -> None:
        expected_code = """
        library ieee;
        use ieee.std_logic_1164.all;
        use ieee.numeric_std.all;
        use ieee.math_real.all;

        entity tanh_tb is
            generic (
                DATA_WIDTH : integer := 16;
                FRAC_WIDTH : integer := 0
            );
            port (
                clk : out std_logic
            );
        end entity tanh_tb;

        architecture rtl of tanh_tb is

            signal clk_period : time := 1 ns;
            signal test_input : signed(16-1 downto 0):=(others=>'0');
            signal test_output : signed(16-1 downto 0);

            component tanh is
                generic (
                    DATA_WIDTH : integer := 16;
                    FRAC_WIDTH : integer := 0
                );
                port (
                    x : in signed(DATA_WIDTH-1 downto 0);
                    y : out signed(DATA_WIDTH-1 downto 0)
                );
            end component tanh;

        begin

            clock_process: process
            begin
                clk <= '0';
                wait for clk_period/2;
                clk <= '1';
                wait for clk_period/2;
            end process clock_process;

            uut: tanh
            port map (
                x => test_input,
                y => test_output
            );

            test_process: process
            begin
                report "======Simulation Start======" severity Note;

                test_input <= to_signed(-1281,16);
                wait for 1*clk_period;
                report "The value of 'test_output' is " & integer'image(to_integer(unsigned(test_output)));
                assert test_output=-256 report "The test case -1281 fail" severity failure;

                test_input <= to_signed(-1000,16);
                wait for 1*clk_period;
                report "The value of 'test_output' is " & integer'image(to_integer(unsigned(test_output)));
                assert test_output=-255 report "The test case -1000 fail" severity failure;

                test_input <= to_signed(-500,16);
                wait for 1*clk_period;
                report "The value of 'test_output' is " & integer'image(to_integer(unsigned(test_output)));
                assert test_output=-246 report "The test case -500 fail" severity failure;

                test_input <= to_signed(0,16);
                wait for 1*clk_period;
                report "The value of 'test_output' is " & integer'image(to_integer(unsigned(test_output)));
                assert test_output=0 report "The test case 0 fail" severity failure;

                test_input <= to_signed(500,16);
                wait for 1*clk_period;
                report "The value of 'test_output' is " & integer'image(to_integer(unsigned(test_output)));
                assert test_output=245 report "The test case 500 fail" severity failure;

                test_input <= to_signed(800,16);
                wait for 1*clk_period;
                report "The value of 'test_output' is " & integer'image(to_integer(unsigned(test_output)));
                assert test_output=254 report "The test case 800 fail" severity failure;

                test_input <= to_signed(1024,16);
                wait for 1*clk_period;
                report "The value of 'test_output' is " & integer'image(to_integer(unsigned(test_output)));
                assert test_output=255 report "The test case 1024 fail" severity failure;


                report "======Simulation Success======" severity Note;
                report "Please check the output message." severity Note;

                wait;

            end process test_process;

        end architecture rtl;
        """
        to_fp = partial(float_values_to_fixed_point, total_bits=16, frac_bits=0)
        tanh = PrecomputedScalarTestBench(
            component_name="tanh",
            x_list_for_testing=to_fp([-1281, -1000, -500, 0, 500, 800, 1024]),
            y_list_for_testing=to_fp([-256, -255, -246, 0, 245, 254, 255]),
        )
        tanh_code = tanh()
        tanh_code_str = ""
        for line in tanh_code:
            tanh_code_str += line + "\n"
        self.check_generated_code(expected_code, tanh_code_str)
