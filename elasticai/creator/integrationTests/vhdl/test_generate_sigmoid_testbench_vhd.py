from elasticai.creator.tests.vhdl.vhdl_file_testcase import GeneratedVHDLCodeTest
from elasticai.creator.vhdl.generator.precomputed_scalar_function import (
    PrecomputedScalarTestBench,
)


class SigmoidTestBenchTest(GeneratedVHDLCodeTest):
    def test_compare_files(self) -> None:
        expected_code = """
        library ieee;
        use ieee.std_logic_1164.all;
        use ieee.numeric_std.all;
        use ieee.math_real.all;

        entity sigmoid_tb is
            generic (
                DATA_WIDTH : integer := 16;
                FRAC_WIDTH : integer := 8
            );
            port (
                clk : out std_logic
            );
        end entity sigmoid_tb;

        architecture rtl of sigmoid_tb is

            signal clk_period : time := 1 ns;
            signal test_input : signed(16-1 downto 0):=(others=>'0');
            signal test_output : signed(16-1 downto 0);

            component sigmoid is
                generic (
                    DATA_WIDTH : integer := 16;
                    FRAC_WIDTH : integer := 8
                );
                port (
                    x : in signed(DATA_WIDTH-1 downto 0);
                    y : out signed(DATA_WIDTH-1 downto 0)
                );
            end component sigmoid;

        begin

            clock_process: process
            begin
                clk <= '0';
                wait for clk_period/2;
                clk <= '1';
                wait for clk_period/2;
            end process clock_process;

            uut: sigmoid
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
                assert test_output=0 report "The test case -1281 fail" severity failure;

                test_input <= to_signed(-1000,16);
                wait for 1*clk_period;
                report "The value of 'test_output' is " & integer'image(to_integer(unsigned(test_output)));
                assert test_output=4 report "The test case -1000 fail" severity failure;

                test_input <= to_signed(-500,16);
                wait for 1*clk_period;
                report "The value of 'test_output' is " & integer'image(to_integer(unsigned(test_output)));
                assert test_output=28 report "The test case -500 fail" severity failure;


                report "======Simulation Success======" severity Note;
                report "Please check the output message." severity Note;

                wait;

            end process test_process;

        end architecture rtl;
        """
        sigmoid = PrecomputedScalarTestBench(
            data_width=16,
            frac_width=8,
            component_name="sigmoid",
            x_list_for_testing=[-1281, -1000, -500],
            y_list_for_testing=[0, 4, 28],
        )
        sigmoid_code = sigmoid()
        sigmoid_code_str = ""
        for line in sigmoid_code:
            sigmoid_code_str += line + "\n"
        self.check_generated_code(expected_code, sigmoid_code_str)
